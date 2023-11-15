import os
import mne
import pandas as pd
from fuzzywuzzy import process

def fuzzy_channel_matching(channels_of_interest, raw_channel_names):
    matched_channels = [process.extractOne(channel, raw_channel_names)[0] for channel in channels_of_interest]
    return matched_channels

def extract_eeg_data(edf_file, channels_of_interest):
    raw = mne.io.read_raw_edf(edf_file, preload=True)

    raw_channel_names = raw.ch_names

    matched_channels = fuzzy_channel_matching(channels_of_interest, raw_channel_names)

    picks = mne.pick_channels(raw.ch_names, include=matched_channels)

    eeg_data = raw.get_data(picks=picks)

    return eeg_data, matched_channels

def save_to_csv(eeg_data, channels_of_interest, output_csv):
    df = pd.DataFrame(eeg_data.T, columns=channels_of_interest)

    df.to_csv(output_csv, index=False)
    print(f'Dane zapisane do {output_csv}')

if __name__ == "__main__":
    patient = 15
    files_list = ["Record1.edf"]
    channels_of_interest = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    data = []

    for file_name in files_list:
        file_path = os.path.join("./Raw_EDF_Files", "p" + str(patient) + "_" + file_name)
        output_csv_path = "Wynik_" + file_name + ".csv"  # Corrected this line
        eeg_data, matched_channels = extract_eeg_data(file_path, channels_of_interest)
        save_to_csv(eeg_data, matched_channels, output_csv_path)
