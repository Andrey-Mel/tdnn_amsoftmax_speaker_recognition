import numpy as np
import librosa
from tensorflow.keras.utils import Sequence, to_categorical



class DataGenerator(Sequence):
    def __init__(self, lst_audio, to_fit=True, batch_size=32, sample_rate=16000, dim=(301, 23),
                 n_classes=1000, shuffle=True):
        self.lst_audio = lst_audio
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.dim = dim
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_lst, self.label_lst = self.audio_3s(self.lst_audio)
        self.len_dataset = len(self.data_lst)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.len_dataset / self.batch_size))

    def __getitem__(self, index):
        x, y = [], []
        indexes = self.indexes[self.batch_size * index: (index + 1) * self.batch_size]
        # data_lst, label_lst = self.features_3s(self.lst_audio)
        for idx in indexes:

            x.extend(self.features_x(self.data_lst[idx]))
            y.append(self.label_lst[idx])

        if self.to_fit:

            return np.array(x), np.array(y)

        else:
            # print('return to fit False')
            return np.array(x)

    def on_epoch_end(self):
        self.indexes = np.arange(self.len_dataset)
        if self.shuffle == True:
            np.random.seed(42)
            np.random.shuffle(self.indexes)

    def preprocess_audio(self, y1):
        # frame the audio signal
        n_frames = int(0.025 * self.sample_rate)
        hop_length = int(0.01 * self.sample_rate)

        # frames = librosa.util.frame(y2,  frame_length=n_frames, hop_length=hop_length)
        mfccs = librosa.feature.mfcc(y=y1, sr=self.sample_rate, n_mfcc=23, n_fft=n_frames, hop_length=hop_length).T
        mfccs -= np.mean(mfccs, axis=0)
        mfccs /= np.std(mfccs, axis=0)

        return mfccs  # [...,None] 301,23

    #@staticmethod
    def get_label(self, path):
        num_class = int(str(path.parts[5]).split('_')[0])
        label = to_categorical(int(num_class), self.n_classes)
        return label

    def features_x(self, sample):  # features_X
        data = []
        len_trek = len(sample)
        # start = 0
        step = 48000
        # parts = int(len_trek / step)
        if len_trek < step:
            sample = np.pad(sample, (0, max(0, step - len_trek), 'constant'))
        sample = sample + 0.009 * np.random.normal(0, 1, step)  # noise
        sample = np.roll(sample, 1600)  # shift
        mfcc = self.preprocess_audio(sample)
        data.append(mfcc)

        return np.array(data)

    def audio_3s(self, audio):
        lst_audio_len_3s, labels = [], []
        for path in self.lst_audio:
            label = self.get_label(path)
            y, sr = librosa.load(path, sr=16000)
            len_trek = len(y)
            start = 0
            step = 48000
            parts = int(len_trek / step)
            for i in range(parts):
                lst_audio_len_3s.append(y[start * i: (start * i) + step])
                labels.append(label)

        return lst_audio_len_3s, labels  # exit features from librosa to 3 second
