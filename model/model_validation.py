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
        data = pd.read_csv('./model/cropped_records/4_training_record.csv').to_numpy().T
        attributes_frames = tf.signal.frame(data, self.frame_size, self.frame_size)
        attributes_frames = tf.math.reduce_mean(tf.math.square(attributes_frames), axis=2)
        attributes_frames = attributes_frames.numpy().T
        attributes_frames= tf.expand_dims(attributes_frames, axis=1, name=None)
        wynik = self.model.predict(attributes_frames)
        # wynik = wynik.round().astype(int)
        df = pd.DataFrame(wynik)
        df.to_csv('predictions.csv')



mv = ModelValidation()
mv.validate()