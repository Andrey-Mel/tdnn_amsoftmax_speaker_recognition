import numpy as np
import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from keras import regularizers, activations, initializers, regularizers, constraints




class AMSoftmax(Layer):
    def __init__(self, units, s, m,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs
                 ):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(AMSoftmax, self).__init__(**kwargs)
        self.units = units
        self.s = s
        self.m = m
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "s": self.s,
            "m": self.m,
            "regularizer": self.kernel_regularizer,
            "initializer": self.kernel_initializer,
            "constraint": self.kernel_constraint

        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        super(AMSoftmax, self).build(input_shape)

        assert len(input_shape) >= 2, 'len(input_shape)>=2'
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      trainable=True,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        #  self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        inputs = tensorflow.nn.l2_normalize(inputs, dim=-1)  # dim=-1
        kernel = tensorflow.nn.l2_normalize(self.kernel, dim=(0, 1))
        #

        dis_cosin = K.dot(inputs, kernel)  # theta for cosine
        psi = dis_cosin - self.m  # phi => (cosineO - m)

        e_costheta = K.exp(self.s * dis_cosin)  # s*(w.T * f(psi)) = > e**s*(cosine(O))
        e_psi = K.exp(self.s * psi)  # e**s*(cosineO) в знаменателе
        sum_x = K.sum(e_costheta, axis=-1, keepdims=True)  # sum e**w.T*f

        temp = e_psi - e_costheta
        temp = temp + sum_x

        output = e_psi / temp

        return output

class ArcFace(Layer):
    def __init__(self, n_classes=100, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_classes": self.n_classes,
            "s": self.s,
            "m": self.m,
            "regularizer": self.regularizer

        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tensorflow.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tensorflow.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = K.dot(x, W)  # x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tensorflow.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tensorflow.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tensorflow.nn.softmax(logits)
        # print(out.shape)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)
