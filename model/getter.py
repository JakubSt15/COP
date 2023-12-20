from datetime import datetime, timedelta

def get_sample(start_register, start_attack, end_attack, hz=512):
    start_register = datetime.strptime(start_register, "%H.%M.%S")
    start_attack = datetime.strptime(start_attack, "%H.%M.%S")
    end_attack = datetime.strptime(end_attack, "%H.%M.%S")
    
    start_attack_sample = (int((start_attack - start_register).total_seconds() * hz))
    end_attack_sample = (int((end_attack - start_register).total_seconds() * hz))

    return start_attack_sample, end_attack_sample

print(get_sample("19.39.33", "19.58.36", "19.59.46"))