import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

path_start = 'model_pytorch/records/'
csv = '5'


df = pd.read_csv(path_start+csv+'_training_record.csv')

def label_column(self, start_attack_sample, end_attack_sample, end_record_sample, no):

    len_attack = end_attack_sample - start_attack_sample
    start_record_sample = max(start_attack_sample - len_attack // 2, 0)  # Adjusted to start from 0
    end_record_sample = min(end_attack_sample + len_attack // 2, end_record_sample)
    labels_training = np.zeros(end_record_sample - start_record_sample)
    labels_training[start_attack_sample - start_record_sample:end_attack_sample - start_record_sample] = 1
    labels_attack = np.zeros(int(end_attack_sample - start_attack_sample))
    labels_attack[0:end_attack_sample-start_attack_sample] = 1

    self.save_mask_to_csv(labels_training, f'./model_pytorch/labels/{no}_training_label.csv')


def get_sample_number(self, reg_start, reg_end, attack_start, attack_end, Hz):
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

    return int(seconds_start_midnight * Hz), int(seconds_end_midnight * Hz), int(seconds_reg_end_midnight * Hz)


time_file = "badaniaDane.txt"

with open(time_file, 'r') as file:
    reader = csv.DictReader(file, delimiter=';')

    for i, row in enumerate(reader):
        file_name = row['File name']
        reg_start = row['Registration start time']
        reg_end = row['Registration end time']
        attack_start = row['Seizure start time']
        attack_end = row['Seizure end time']

        start, end, end_sec = get_sample_number(reg_start, reg_end, attack_start, attack_end, 512)
        label_column(start, end, end_sec, i+1)