import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import signal
import mne
from normalization import Normalization


class ModelValidation:
    def __init__(self):
        self.model = tf.keras.models.load_model('model.keras')
        self.validation_file = 'p10_Record2.edf'
        self.frame_size = 1000
        self.normalization = Normalization()
        self.normalization.min = np.loadtxt("min.out")
        self.normalization.max = np.loadtxt("max.out")
        self.selected_ch = ['EEG Fp1-Ref', 'EEG Fp2-Ref', 'EEG F3-Ref', 'EEG F4-Ref',
                            'EEG C3-Ref', 'EEG C4-Ref', 'EEG P3-Ref', 'EEG P4-Ref',
                            'EEG O1-Ref', 'EEG O2-Ref', 'EEG F7-Ref', 'EEG F8-Ref',
                            'EEG T3-Ref', 'EEG T4-Ref', 'EEG T5-Ref', 'EEG T6-Ref',
                            'EEG Fz-Ref', 'EEG A1-Ref', 'EEG A2-Ref']

    def validate(self):
        edf = mne.io.read_raw_edf(self.normalization)
        ind_of_channels = mne.pick_channels(edf.ch_names, include=self.selected_ch)
        data = edf.get_data()
        data = data[ind_of_channels, :self.frame_size * 2]
        data = self.butter_filter(data, 3, 25, edf.info["sfreq"])
        data = self.normalization.transform2d(data)
        data = data.T
        data = data[np.newaxis, self.frame_size:self.frame_size * 2, :]
        wynik = self.model.predict(data)
        self.show_differences(data, wynik, 1)

    def butter_filter(self, raw_data, low_cut, high_cut, frequency):
        b, a = signal.butter(N=4, Wn=[low_cut, high_cut], btype='band', fs=frequency, analog=False)
        for channel in range(19):
            raw_data[channel] = signal.lfilter(b, a, raw_data[channel])
        return raw_data

    def show_differences(self, data, wynik, kanal):
        plt.plot(data[2000:3000, kanal])
        plt.title("oryginal kanal: " + str(kanal))
        plt.plot(wynik[0, :, kanal])
        plt.title("predykcja kanal: " + str(kanal))
        plt.show()


mv = ModelValidation()
mv.validate()
