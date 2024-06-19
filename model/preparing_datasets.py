# import mne
import numpy as np
import tensorflow as tf
import pandas as pd
from normalization import Normalization
from filter import Filter

fx = './model/train_data/mask_attack_'
fy = './model/cropped_records/'
class PreparingDatasets:
    def __init__(self):
        self.selected_ch = ['EEG Fp1-Ref', 'EEG Fp2-Ref', 'EEG F3-Ref', 'EEG F4-Ref',
                            'EEG C3-Ref', 'EEG C4-Ref', 'EEG P3-Ref', 'EEG P4-Ref',
                            'EEG O1-Ref', 'EEG O2-Ref', 'EEG F7-Ref', 'EEG F8-Ref',
                            'EEG T3-Ref', 'EEG T4-Ref', 'EEG T5-Ref', 'EEG T6-Ref',
                            'EEG Fz-Ref', 'EEG A1-Ref', 'EEG A2-Ref']
        self.label_files = [f'{fx}1.csv',f'{fx}2.csv',  f'{fx}4.csv']
        self.train_files = [
        f'{fy}1_training_record.csv',
        f'{fy}2_training_record.csv',
        f'{fy}4_training_record.csv'
        ]
        self.frequency = 512
        self.frame_size = 100
        self.normalization = Normalization()
        self.filter=Filter()

    def shift_edf(self, edf_data, shift):
        data_shifted = edf_data.iloc[:, shift:].values
        data_shifted = data_shifted[:, :data_shifted.shape[1] - (data_shifted.shape[1] % self.frame_size)]
        data_shifted = np.split(data_shifted[:, :int(data_shifted.shape[1] / self.frame_size) * self.frame_size],
                                int(data_shifted.shape[1] / self.frame_size), axis=1)
        data_shifted = np.array(data_shifted)
        data_shifted = np.transpose(data_shifted, (0, 2, 1))
        return tf.convert_to_tensor(data_shifted)

    def read_and_prepare_data(self, train_files, label_files, normalizator):
        x_file = train_files
        y_file = label_files
        df_x = pd.read_csv(x_file).T
        df_y = pd.read_csv(y_file).T

        data = self.filter.butter_filter(df_x, self.frequency)
        data = normalizator.transform2d(data)
        file = x_file[:-4]
        x = self.shift_edf(data, 0)

        y = self.shift_edf(df_y, 0)
        x = x[:y.shape[0] - 1, :, :19]
        y = y[:-1, :, :19]
        print("zapisuje plik: " + file)
        tf.io.write_file(file + "_x", tf.io.serialize_tensor(x))
        tf.io.write_file(file + "_y", tf.io.serialize_tensor(y))

    def generate_datasets(self):
        self.normalization.fit_all(self.train_files)
        np.savetxt('min.out', self.normalization.min, delimiter=',')
        np.savetxt('max.out', self.normalization.max, delimiter=',')
        for file_number in range(len(self.train_files)):
            self.read_and_prepare_data(self.train_files[file_number], self.label_files[file_number], self.normalization)


pr = PreparingDatasets()
pr.generate_datasets()
