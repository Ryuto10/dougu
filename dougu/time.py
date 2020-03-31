from datetime import datetime


def get_time(readable=True):
    if readable:
        return datetime.today().strftime("%Y/%m/%d %H:%M:%S")
    else:
        return datetime.today().strftime("%m%d%H%M")
