import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fuzzywuzzy import process

class EEGAnalyzer:
    def __init__(self, file_path, channels_of_interest, step=1, threshold=0):
        self.file_path = file_path
        self.channels_of_interest = channels_of_interest
        self.step = step
        self.threshold = threshold
        self.output_csv_path = f"Array_of_{os.path.basename(file_path)}.csv"
        self.eeg_data, self.matched_channels, self.sfreq, self.raw = self.extract_eeg_data()

    def fuzzy_channel_matching(self, raw_channel_names):
        matched_channels = [process.extractOne(channel, raw_channel_names)[0] for channel in self.channels_of_interest]
        return matched_channels

    def extract_eeg_data(self):
        raw = mne.io.read_raw_edf(self.file_path, preload=True)

        raw_channel_names = raw.ch_names

        matched_channels = self.fuzzy_channel_matching(raw_channel_names)

        picks = mne.pick_channels(raw.ch_names, include=matched_channels)

        eeg_data = raw.get_data(picks=picks)

        return eeg_data, matched_channels, raw.info['sfreq'], raw

    def generate_seizure_mask(self):
        num_seconds = int(self.eeg_data.shape[1] / self.sfreq)

        seizure_mask = np.zeros(num_seconds // self.step, dtype=int)

        for i in range(0, num_seconds, self.step):
            if np.any(self.eeg_data[:, int(i * self.sfreq):(int((i + 1) * self.sfreq))] >= self.threshold):
                seizure_mask[i // self.step] = 1

        return seizure_mask

    def save_mask_to_csv(self, mask):
        df = pd.DataFrame({'Seizure_Mask': mask})
        df.to_csv(self.output_csv_path, index=False)

    def plot_eeg_data(self):
        times = self.raw.times
        seizure_mask = np.zeros(len(times), dtype=int)

        for i in range(len(times)):
            if np.any(self.eeg_data[:, i] >= self.threshold):
                seizure_mask[i] = 1

        plt.plot(times, self.eeg_data.T)
        plt.xlabel('Time (s)')
        plt.ylabel('EEG Data')
        plt.legend(self.matched_channels)

        plt.twinx()
        plt.plot(times, seizure_mask, 'r', alpha=0.5)
        plt.ylabel('Seizure Mask')

        plt.show()

if __name__ == "__main__":
    files_list = ["PN00-1.edf"]
    channels_of_interest = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz',
                            'P4', 'T6', 'O2']

    for file_name in files_list:
        analyzer = EEGAnalyzer(file_name, channels_of_interest, step=1, threshold=0.00075) #step=co jaki krok ma zapisywać threshold=od jakiej wartości(>=) ma szukać
        seizure_mask = analyzer.generate_seizure_mask()
        analyzer.save_mask_to_csv(seizure_mask)
        #analyzer.plot_eeg_data()  Dużo czasu zajmuje a nie jest niezbędne