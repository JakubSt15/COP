import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import signal
# import mne
from normalization import Normalization
from filter import Filter
import pandas as pd
class ModelValidation:
    def __init__(self):
        self.model = tf.keras.models.load_model('model.keras')
        self.validation_file = 'p10_Record2.edf'
        self.frame_size = 1000
        self.normalization = Normalization()
        self.filter=Filter()
        self.normalization.min = np.loadtxt("min.out")
        self.normalization.max = np.loadtxt("max.out")
        self.selected_ch = ['EEG Fp1-Ref', 'EEG Fp2-Ref', 'EEG F3-Ref', 'EEG F4-Ref',
                            'EEG C3-Ref', 'EEG C4-Ref', 'EEG P3-Ref', 'EEG P4-Ref',
                            'EEG O1-Ref', 'EEG O2-Ref', 'EEG F7-Ref', 'EEG F8-Ref',
                            'EEG T3-Ref', 'EEG T4-Ref', 'EEG T5-Ref', 'EEG T6-Ref',
                            'EEG Fz-Ref', 'EEG A1-Ref', 'EEG A2-Ref']

    def validate(self):
        data = pd.read_csv('./model/train_data/_1.csv')
        data2 = pd.read_csv('./_1_attack.csv')
        data = self.filter.butter_filter(data,  512)
        data = self.normalization.transform2d(data)
        data = data.T
        data = np.array(data).T
        print(data.shape)
        data = data[np.newaxis, :self.frame_size, :]
        wynik = self.model.predict(data)
        print(wynik.shape)
        df = pd.DataFrame(wynik[0])
        df.to_csv('predictions.csv')



mv = ModelValidation()
mv.validate()