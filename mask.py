import re
import os


katalog = "BadanieDane/"

files_list = os.listdir(katalog)
#print(files_list)


for file_name in files_list:
    #file_path = os.path.join("./Baza", file_name)
    with open("./BadanieDane/" + file_name, "r") as file:
        data = file.read()

    pattern = re.compile(r"""
         Seizure\sn\s\d+\s
         File\sname:\s\w+-(\d+)\.edf\s
         Registration\sstart\stime:\s(\d+\.\d+\.\d+)\s
         Registration\send\stime:\s(\d+\.\d+\.\d+)\s
         Seizure\sstart\stime:\s(\d+\.\d+\.\d+)\s
         Seizure\send\stime:\s(\d+\.\d+\.\d+)
        """, re.VERBOSE)

    matches = pattern.findall(data)

    seizure_data = []

    print('matches ',matches)
    print('matches len ',len(matches))


    for match in matches:
        seizure_number, reg_start, reg_end, seizure_start, seizure_end = match
        print(f"Seizure n {seizure_number}:")
        print(f"  Registration Start Time: {reg_start}")
        print(f"  Registration End Time: {reg_end}")
        print(f"  Seizure Start Time: {seizure_start}")
        print(f"  Seizure End Time: {seizure_end}")

        # Calculate time for each seizure
        reg_start_time = sum(int(x) * 60 ** i for i, x in enumerate(reversed(reg_start.split('.'))))
        reg_end_time = sum(int(x) * 60 ** i for i, x in enumerate(reversed(reg_end.split('.'))))
        seizure_start_time = sum(int(x) * 60 ** i for i, x in enumerate(reversed(seizure_start.split('.'))))
        #seizure_end_time = sum(int(x) * 60 ** i for i, x in enumerate(reversed(seizure_end.split('.'))))
        seizure_end_time = min(sum(int(x) * 60 ** i for i, x in enumerate(reversed(seizure_end.split('.')))), reg_end_time)

        seizure_array = [0] * (reg_end_time - reg_start_time + 1)

        for i in range(seizure_start_time - reg_start_time, seizure_end_time - reg_start_time + 1):
            seizure_array[i] = 1

        seizure_data.append((seizure_number, seizure_array))
        print()

        text_file_path = file_name + "_seizure_array_data.txt"
        print(file_name)
        with open(text_file_path, "w") as text_file:
            for seizure_number, seizure_array in seizure_data:
                text_file.write(f"Seizure n {seizure_number} Array:\n")
                text_file.write(",".join(map(str, seizure_array)) + "\n\n")

        print(f"\nArray data written to: {text_file_path}")

        for seizure_number, seizure_array in seizure_data:
            print(f"Seizure n {seizure_number} Array:")
            print(seizure_array,"\n")


#Ten przypadek chyba źle jest zapisany, rejestracja kończy się przed zakończeniem ataku???
#W gruncie rzeczy nie utworzy z tego tablicy

#Seizure n 3
#File name: PN00-3.edf
#Registration start time: 18.15.44
#Registration end time: 18.57.13
#Seizure start time: 18.28.29
#Seizure end time: 19.29.29

