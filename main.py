import csv
import os
import mne
import matplotlib.pyplot as plt
from fuzzywuzzy import process
import pandas as pd
from datetime import datetime, timedelta

class EEGProcessorAndPlotter:
    def __init__(self, file_path, time_file):
        self.file_path = file_path
        self.time_file = time_file
        self.raw = self.load_raw_data()
        self.channels_of_interest = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']

    def load_raw_data(self):
        raw = mne.io.read_raw_edf(self.file_path, preload=True)
        return raw

    def process_eeg_data(self):
        raw_channel_names = self.raw.ch_names
        matched_channels = self.fuzzy_channel_matching(raw_channel_names)
        picks = mne.pick_channels(raw_channel_names, include=matched_channels)
        eeg_data = self.raw.get_data(picks=picks) * 1000000
        print("raw_channel_names: ", raw_channel_names, "\n",
              "matched_channels: ", matched_channels)
        return eeg_data, self.raw.times, self.raw.info['sfreq']

    def plot_eeg_channels(self):
        eeg_data, times, sfreq = self.process_eeg_data()
        num_channels = len(self.channels_of_interest)

        if num_channels > 1:
            fig, axs = plt.subplots(num_channels, 1, figsize=(10, 5 * num_channels))
            for i, channel_name in enumerate(self.channels_of_interest):
                axs[i].plot(times, eeg_data[i, :].T, label=channel_name)
                axs[i].set_xlabel('Time (s)')
                axs[i].set_ylabel('EEG Data')
                axs[i].legend()
        elif num_channels == 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(times, eeg_data[0, :].T, label=self.channels_of_interest[0])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('EEG Data')
            ax.legend()

        plt.tight_layout()
        plt.show()

    def fuzzy_channel_matching(self, raw_channel_names):
        matched_channels = []
        for channel in self.channels_of_interest:
            matched_channel, score = process.extractOne(channel, raw_channel_names)
            if score < 90:  # Próg oceny dopasowania
                print(
                    f"Nie można znaleźć dokładnego dopasowania dla kanału {channel}. Najlepsze dopasowanie to {matched_channel} z wynikiem {score}.")
            matched_channels.append(matched_channel)
        return matched_channels

    def save_eeg_data_to_csv(self, output_file):
        eeg_data, _, _ = self.process_eeg_data()

        df = pd.DataFrame(eeg_data.T, columns=self.channels_of_interest)

        df = df[self.channels_of_interest]

        df.to_csv(output_file, sep=',', index=False)

        print("Data saved to", output_file)

    def save_mask_to_csv(self, mask, path=None):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        path = self.output_csv_path if path is None else path
        df = pd.DataFrame(mask)
        df.to_csv(path, index=False)

    def get_attack_csv(self, start_attack_sample, end_attack_sample, end_record_sample, no):
        eeg_data, _, _ = self.process_eeg_data()

        len_attack = end_attack_sample - start_attack_sample
        start_record = max(start_attack_sample - len_attack // 2, 0)  # Adjusted to start from 0
        end_record = min(end_attack_sample + len_attack // 2, end_record_sample)
        print("end_rcord_sample", end_record_sample)
        print("end_record", end_record)
        seizure_mask = self.channel_mask_prepare()
        training_mask = self.channel_mask_prepare()

        for j, channel in enumerate(self.channels_of_interest):
            channel_mask = []
            channel_mask_training = []
            for i in range(start_attack_sample, end_attack_sample):
                if i < eeg_data.shape[1]:
                    channel_mask.append(eeg_data[j, i])

            for i in range(start_record, end_record):
                if i < eeg_data.shape[1]:
                    channel_mask_training.append(eeg_data[j, i])
            seizure_mask[channel] = channel_mask
            training_mask[channel] = channel_mask_training

        self.save_mask_to_csv(seizure_mask, f'./masks/{no}_only_attack.csv')
        self.save_mask_to_csv(training_mask, f'./masks/{no}_training_record.csv')

    def get_sample_number(self, reg_start, reg_end, attack_start, attack_end, Hz, output_file):
        reg_start = datetime.strptime(reg_start, "%H.%M.%S")
        reg_end = datetime.strptime(reg_end, "%H.%M.%S")
        start = datetime.strptime(attack_start, "%H.%M.%S")
        end = datetime.strptime(attack_end, "%H.%M.%S")

        if start < reg_start:
            start += timedelta(hours=24)

        start = reg_start.replace(hour=start.hour, minute=start.minute, second=start.second)
        end = reg_start.replace(hour=end.hour, minute=end.minute, second=end.second)
        seconds_start_midnight = int((start - reg_start).total_seconds())
        seconds_end_midnight = int((end - reg_start).total_seconds())

        print("times", seconds_start_midnight, seconds_end_midnight)
        if reg_end < reg_start:
            reg_end += timedelta(hours=24)

        seconds_reg_end_midnight = int((reg_end - reg_start).total_seconds())

        if seconds_start_midnight < 0:
            seconds_start_midnight += 24 * 3600
        if seconds_end_midnight < 0:
            seconds_end_midnight += 24 * 3600

        output_text = f"{seconds_start_midnight * Hz} {seconds_end_midnight * Hz}"

        with open(output_file, 'a') as file:
            file.write(output_text + '\n')

        print(output_text)

        return int(seconds_start_midnight * Hz), int(seconds_end_midnight * Hz), int(seconds_reg_end_midnight * Hz)

    def channel_mask_prepare(self):
        channel_mask = {}
        for channel in self.channels_of_interest:
            channel_mask[channel] = []
        return channel_mask

time_file = "badaniaDane_Test.txt"

with open(time_file, 'r') as file:
    reader = csv.DictReader(file, delimiter=';')

    for i, row in enumerate(reader):
        file_name = row['File name']
        reg_start = row['Registration start time']
        reg_end = row['Registration end time']
        attack_start = row['Seizure start time']
        attack_end = row['Seizure end time']

        eeg_processor_plotter = EEGProcessorAndPlotter(file_name, time_file)
        eeg_processor_plotter.plot_eeg_channels()
        eeg_processor_plotter.save_eeg_data_to_csv(f"./Entire_Sample/all_channel_samples_{i}.csv")
        output_file_path = f"{os.path.splitext(file_name)[0]}_output.csv"

        start, end, end_sec = eeg_processor_plotter.get_sample_number(reg_start, reg_end, attack_start, attack_end, 512, output_file_path)
        eeg_processor_plotter.get_attack_csv(start, end, end_sec, i + 1)
