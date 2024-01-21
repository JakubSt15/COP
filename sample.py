from datetime import datetime, timedelta

def get_sample_number(start, end, search_start_attack,search_end_attack,Hz):
    reg_start = datetime.strptime(start, "%H.%M.%S")
    reg_end = datetime.strptime(end, "%H.%M.%S")

    start = datetime.strptime(search_start_attack, "%H.%M.%S")
    end = datetime.strptime(search_end_attack, "%H.%M.%S")
    seconds_start = int((start - reg_start).total_seconds())
    seconds_end = int((end - reg_start).total_seconds())
    end_sample = int()
    return seconds_start*Hz, seconds_end*Hz

start,end=get_sample_number("18.15.44","18.28.29","19.29.29", 512)

print(start,end)