import mne
import numpy as np
import tensorflow as tf
from scipy import signal
from normalization import Normalization


class PreparingDatasets:
    def __init__(self):
        self.selected_ch = ['EEG Fp1-Ref', 'EEG Fp2-Ref', 'EEG F3-Ref', 'EEG F4-Ref',
                            'EEG C3-Ref', 'EEG C4-Ref', 'EEG P3-Ref', 'EEG P4-Ref',
                            'EEG O1-Ref', 'EEG O2-Ref', 'EEG F7-Ref', 'EEG F8-Ref',
                            'EEG T3-Ref', 'EEG T4-Ref', 'EEG T5-Ref', 'EEG T6-Ref',
                            'EEG Fz-Ref', 'EEG A1-Ref', 'EEG A2-Ref']
        self.train_files = ['p10_Record1.edf', 'p10_Record2.edf', 'p11_Record1.edf', 'p11_Record2.edf',
                            'p11_Record3.edf',
                            'p11_Record4.edf',
                            'p12_Record1.edf', 'p12_Record2.edf', 'p12_Record3.edf', 'p13_Record1.edf',
                            'p13_Record2.edf',
                            'p13_Record3.edf', 'p13_Record4.edf', 'p14_Record1.edf', 'p14_Record2.edf',
                            'p14_Record3.edf',
                            'p15_Record1.edf', 'p15_Record2.edf', 'p15_Record3.edf', 'p15_Record4.edf']
        self.frequency = 500
        self.frame_size = 1000
        self.split_ratio = 0.8
        self.batch_size = 50
        self.normalization = Normalization()

    def shift_edf(self, edf_data, shift):
        data_shifted = edf_data[:, shift:]
        data_shifted = data_shifted[:, :data_shifted.shape[1] - (data_shifted.shape[1] % self.frame_size)]
        data_shifted = np.split(data_shifted[:, :int(data_shifted.shape[1] / self.frame_size) * self.frame_size],
                                int(data_shifted.shape[1] / self.frame_size), axis=1)
        data_shifted = np.array(data_shifted)
        data_shifted = np.transpose(data_shifted, (0, 2, 1))
        return tf.convert_to_tensor(data_shifted)

    def read_and_prepare_data(self, file, normalizator):
        edf = mne.io.read_raw_edf(file, preload=True)
        ind_of_channels = mne.pick_channels(edf.ch_names,
                                            include=self.selected_ch)
        data = edf.get_data()
        data = data[ind_of_channels]
        data = self.butter_filter(data, 3, 25, edf.info["sfreq"])
        data = normalizator.transform2d(data)
        file = file[:-4]
        for i in range(0, 500, 100):  # TU ZMIENIONE : NA 2 BO ZAJMUJE ZA DUŻO RAMU
            x = self.shift_edf(data, i)
            y = self.shift_edf(data, self.frame_size + i)
            x = x[:y.shape[0] - 1, :, :19]
            y = y[:-1, :, :19]
            print("zapisuje plik: " + file)
            tf.io.write_file(file + "_x_shift_" + str(i), tf.io.serialize_tensor(x))
            tf.io.write_file(file + "_y_shift_" + str(i), tf.io.serialize_tensor(y))

    def butter_filter(self, raw_data, low_cut, high_cut, frequency):
        b, a = signal.butter(N=4, Wn=[low_cut, high_cut], btype='band', fs=frequency, analog=False)
        for channel in range(19):
            raw_data[channel] = signal.lfilter(b, a, raw_data[channel])
        return raw_data

    def generate_datasets(self):
        self.normalization.fit_all(self.train_files)
        np.savetxt('min.out', self.normalization.min, delimiter=',')
        np.savetxt('max.out', self.normalization.max, delimiter=',')
        for file in self.train_files:
            self.read_and_prepare_data(file, self.normalization)


pr = PreparingDatasets()
pr.generate_datasets()
