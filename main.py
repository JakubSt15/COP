import os
import mne
import pandas as pd
import pyarrow.feather as feather
from fuzzywuzzy import process
import re

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

def save_to_feather(eeg_data, channels_of_interest, output_feather):
    df = pd.DataFrame(eeg_data.T, columns=channels_of_interest)

    feather.write_feather(df, output_feather)

    print(f'Dane zapisane do {output_feather}')

if __name__ == "__main__":

    katalog = "Baza/"

    files_list = os.listdir(katalog)
    #print(files_list)
    channels_of_interest = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    data = []

    for file_name in files_list:
        file_path = os.path.join("./Baza", file_name)
        eeg_data, matched_channels = extract_eeg_data(file_path, channels_of_interest)
        output_feather = file_name + ".feather"
        save_to_feather(eeg_data,channels_of_interest,output_feather)
        print(eeg_data)

    # # Ścieżka do katalogu, który chcesz przeszukać
    # files_list = "BadanieDane/"
    #
    # # Pętla przeglądająca pliki w katalogu
    # for file_name in os.listdir(files_list):
    #     if file_name.endswith(".txt"):
    #         # Pełna ścieżka do pliku
    #         full_path = os.path.join(files_list, file_name)
    #
    #         import re
    #
    #         # Tworzenie pustej listy na dane
    #         wyniki = []
    #
    #         # Otwieranie pliku
    #         with open(full_path, 'r') as file:
    #             # Pętla przeglądająca linie w pliku
    #             for line in file:
    #                 # Tutaj możesz przetwarzać zawartość linii, na przykład szukać konkretnego tekstu
    #                 if 'File name:' in line:
    #                     # Wyrażenie regularne do wyszukiwania elementów zaczynających się od "PN" i kończących się na ".edf"
    #                     wyrazenie_regularne = re.compile(r'PN.*\.edf')
    #                     # Wyszukaj dopasowanie w linii tekstu
    #                     dopasowanie = wyrazenie_regularne.search(line)
    #                     if dopasowanie:
    #                         wyniki.append(dopasowanie.group())
    #
    #                 if 'Seizure start time' in line:
    #                     # Wyrażenie regularne do wyszukania godziny w formacie HH.MM.SS
    #                     wyrazenie_regularne = re.compile(r'(\d{2}.\d{2}.\d{2})')
    #                     # Wyszukiwanie dopasowania w linii tekstu
    #                     dopasowanie = wyrazenie_regularne.search(line)
    #                     if dopasowanie:
    #                         wyniki.append(dopasowanie.group(1))
    #
    #                 if 'Seizure end time:' in line:
    #                     # Wyrażenie regularne do wyszukania godziny w formacie HH.MM.SS
    #                     wyrazenie_regularne = re.compile(r'(\d{2}.\d{2}.\d{2})')
    #                     # Wyszukiwanie dopasowania w linii tekstu
    #                     dopasowanie = wyrazenie_regularne.search(line)
    #                     if dopasowanie:
    #                         wyniki.append(dopasowanie.group(1))
    #
    #         # Wyświetlanie zawartości tablicy
    #         print(wyniki)



