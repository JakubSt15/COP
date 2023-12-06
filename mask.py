import re
import os
import csv
from datetime import datetime, timedelta

with open("badaniaDanetest.csv", "r") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=';')

    for row in csv_reader:
        file_name = row['File name']
        reg_start_str = row['Registration start time']
        reg_end_str = row['Registration end time']
        seizure_start_str = row['Seizure start time']
        seizure_end_str = row['Seizure end time']

        try:
            # Convert string times to datetime objects
            reg_start = datetime.strptime(reg_start_str, "%H.%M.%S")
            reg_end = datetime.strptime(reg_end_str, "%H.%M.%S")
            seizure_start = datetime.strptime(seizure_start_str, "%H.%M.%S")
            seizure_end = datetime.strptime(seizure_end_str, "%H.%M.%S")

        except ValueError as e:
            print(f"Error processing file {file_name}: {e}")
            print(f"  Registration Start Time: {reg_start_str}")
            print(f"  Registration End Time: {reg_end_str}")
            print(f"  Seizure Start Time: {seizure_start_str}")
            print(f"  Seizure End Time: {seizure_end_str}")
            continue

        # Adjust times that cross midnight
        if reg_start > reg_end:
            reg_end += timedelta(days=1)
        if seizure_start > seizure_end:
            seizure_end += timedelta(days=1)

        print(f"File Name: {file_name}")
        print(f"  Registration Start Time: {reg_start.strftime('%H:%M:%S')}")
        print(f"  Registration End Time: {reg_end.strftime('%H:%M:%S')}")
        print(f"  Seizure Start Time: {seizure_start.strftime('%H:%M:%S')}")
        print(f"  Seizure End Time: {seizure_end.strftime('%H:%M:%S')}")

        # Calculate time in seconds
        reg_start_time = int((reg_start - datetime.min).total_seconds())
        reg_end_time = int((reg_end - datetime.min).total_seconds())
        seizure_start_time = int((seizure_start - datetime.min).total_seconds())
        seizure_end_time = min(int((seizure_end - datetime.min).total_seconds()), reg_end_time)

        # Calculate array indices
        seizure_array_start = max(0, seizure_start_time - reg_start_time)
        seizure_array_end = min(reg_end_time - reg_start_time, seizure_end_time - reg_start_time)

        # Create and populate seizure_array
        seizure_array = [0] * (reg_end_time - reg_start_time + 1)
        for i in range(seizure_array_start, seizure_array_end + 1):
            seizure_array[i] = 1

        text_file_path = f"SeizureMask/{file_name}_array_data.txt"
        with open(text_file_path, "w") as text_file:
            text_file.write(f"Array for {file_name}:\n")
            text_file.write(",".join(map(str, seizure_array)) + "\n\n")

        print(f"Array data written to: {text_file_path}\n")
