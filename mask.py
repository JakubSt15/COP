import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fuzzywuzzy import process
from datetime import datetime, timedelta
import sys


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

        seizure_mask = {
            'Fp1': [],
            'Fp2': [],
            'F7': [],
            'F3': [],
            'Fz': [],
            'F4': [],
            'F8': [],
            'T3': [],
            'C3': [],
            'Cz': [],
            'C4': [],
            'T4': [],
            'T5': [],
            'P3': [],
            'Pz': [],
            'P4': [],
            'T6': [],
            'O1': [],
            'O2': []
        }

        for j, channel in enumerate(self.channels_of_interest):
            channel_mask = []
            for i in range(0, num_seconds, self.step):
                window_data = self.eeg_data[j, int(i * self.sfreq):int((i + 1) * self.sfreq)]
                if np.any(window_data >= self.threshold):
                    channel_mask.append(1)
                else:
                    channel_mask.append(0)

            seizure_mask[channel] = channel_mask

        # Ensure all channels have the same length, filling missing entries with zeros
        max_length = max(len(seizure_mask[channel]) for channel in seizure_mask)
        for channel in seizure_mask:
            seizure_mask[channel] += [0] * (max_length - len(seizure_mask[channel]))

        return seizure_mask

    def save_mask_to_csv(self, mask, path=None):
        # Ensure the directory exists or create it
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Use pandas DataFrame to save the mask as a CSV file
        path = self.output_csv_path if path is None else path
        df = pd.DataFrame(mask)
        df.to_csv(path, index=False)

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

        # plt.show()

    def channel_mask_prepare(self):
        return {
            'Fp1': [],
            'Fp2': [],
            'F7': [],
            'F3': [],
            'Fz': [],
            'F4': [],
            'F8': [],
            'T3': [],
            'C3': [],
            'Cz': [],
            'C4': [],
            'T4': [],
            'T5': [],
            'P3': [],
            'Pz': [],
            'P4': [],
            'T6': [],
            'O1': [],
            'O2': []
        }

    def get_attack_csv(self, start_attack_sample, end_attack_sample, end_record_sample, no):
        len_attack = end_attack_sample - start_attack_sample
        start_record = max(start_attack_sample - len_attack // 2, 0)  # Adjusted to start from 0
        end_record = min(end_attack_sample + len_attack // 2, end_record_sample)

        seizure_mask = self.channel_mask_prepare()
        training_mask = self.channel_mask_prepare()

        for j, channel in enumerate(self.channels_of_interest):
            channel_mask = []
            channel_mask_training = []
            for i in range(start_attack_sample, end_attack_sample):
                if i < self.eeg_data.shape[1]:  # Added a check to ensure the index is within bounds
                    channel_mask.append(self.eeg_data[j, i])

            for i in range(start_record, end_record):
                if i < self.eeg_data.shape[1]:  # Added a check to ensure the index is within bounds
                    channel_mask_training.append(self.eeg_data[j, i])
            seizure_mask[channel] = channel_mask
            training_mask[channel] = channel_mask_training

        # Save the masks using CSV
        self.save_mask_to_csv(seizure_mask, f'./masks/{no}_only_attack.csv')
        self.save_mask_to_csv(training_mask, f'./masks/{no}_training_record.csv')


# directory = './'

files_list = ["PN00-1.edf"]
channels_of_interest = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz',
                        'P4', 'T6', 'O1', 'O2']


# for file_name in files_list:
#    analyzer = EEGAnalyzer(file_name, channels_of_interest, step=1, threshold=0.00075) #step=co jaki krok ma zapisywać threshold=od jakiej wartości(>=) ma szukać
#    #self.eeg_data, self.matched_channels, self.sfreq, self.raw
#    print(len(analyzer.eeg_data))
#    print(len(analyzer.matched_channels))
#    print(analyzer.sfreq)
#    print(len(analyzer.raw))
#    analyzer.get_attack_csv()
#
# seizure_mask = analyzer.generate_seizure_mask()
# analyzer.save_mask_to_csv(seizure_mask)
# analyzer.plot_eeg_data()  #Dużo czasu zajmuje a nie jest niezbędne

# pogrupowanie poprzez klastry, użyć modelu

def get_sample_number(reg_start, reg_end, attack_start, attack_end, Hz, output_file):
    reg_start = datetime.strptime(reg_start, "%H.%M.%S")
    reg_end = datetime.strptime(reg_end, "%H.%M.%S")

    start = datetime.strptime(attack_start, "%H.%M.%S")
    end = datetime.strptime(attack_end, "%H.%M.%S")

    # Ensure the attack time is within the same day as the registration
    start = reg_start.replace(hour=start.hour, minute=start.minute, second=start.second)
    end = reg_start.replace(hour=end.hour, minute=end.minute, second=end.second)

    if start < reg_start:
        # Add 24 hours to the attack time to count from midnight to the end of the day
        start += timedelta(hours=24)

    # Calculate the time of the attack from midnight
    seconds_start_midnight = int((start - reg_start).total_seconds())
    seconds_end_midnight = int((end - reg_start).total_seconds())

    # If the attack time crosses midnight, add 24 hours to the calculated time
    if end < start:
        seconds_end_midnight += 24 * 60 * 60  # 24 hours in seconds

    output_text = f"{seconds_start_midnight * Hz} {seconds_end_midnight * Hz}"

    # Append to the file instead of overwriting
    with open(output_file, 'a') as file:
        file.write(output_text + '\n')

    print(output_text)

    return int(seconds_start_midnight * Hz), int(seconds_end_midnight * Hz), int((reg_end - reg_start).total_seconds() * Hz)

    # Append to the file instead of overwriting
    file_name = os.path.splitext(os.path.basename(output_file))[0]
    output_text = f"{file_name}:{seconds_start_midnight * Hz} {seconds_end_midnight * Hz}"

    # Append to the file instead of overwriting
    with open(output_file, 'a') as file:
        file.write(output_text + '\n')

    print(output_text)

    return int(seconds_start_midnight * Hz), int(seconds_end_midnight * Hz), int((reg_end - reg_start).total_seconds() * Hz)


# Assuming badaniaDane.txt is in the same directory as the script
file_path = "badaniaDane.txt"
output_file_path = "output.txt"
df = pd.read_csv(file_path, delimiter=';')

channels_of_interest = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz',
                        'P4', 'T6', 'O1', 'O2']

# Create a single output file
with open(output_file_path, 'w') as output_file:
    for i, row in df.iterrows():
        # Execute the EEG analysis for each file
        anylzer = EEGAnalyzer(row['File name'], channels_of_interest, step=1, threshold=0.00075)

        # Get the sample numbers and save the masks
        start, end, end_sec = get_sample_number(row['Registration start time'], row['Registration end time'],
                                                row['Seizure start time'], row['Seizure end time'], 512, output_file_path)

        # Update the EEGAnalyzer instance with the new parameters
        anylzer.channels_of_interest = channels_of_interest
        anylzer.step = 1
        anylzer.threshold = 0.00075
        anylzer.output_csv_path = f"Array_of_{os.path.basename(anylzer.file_path)}.csv"

        # Perform the analysis using the existing 'anylzer'
        anylzer.get_attack_csv(start, end, end_sec, i + 1)
