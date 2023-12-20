from datetime import datetime, timedelta

def get_sample_number(start,search_start_attack,search_end_attack,Hz):
    reg_start = datetime.strptime(start, "%H.%M.%S")
    start = datetime.strptime(search_start_attack, "%H.%M.%S")
    end = datetime.strptime(search_end_attack, "%H.%M.%S")
    seconds_start = int((start - reg_start).total_seconds())
    seconds_end = int((end - reg_start).total_seconds())

    return seconds_start*Hz, seconds_end*Hz

print(get_sample_number("19.39.33","19.58.36","19.59.46", 512))
