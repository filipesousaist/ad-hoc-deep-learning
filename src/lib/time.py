SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 60 * SECONDS_IN_MINUTE
SECONDS_IN_DAY = 24 * SECONDS_IN_HOUR

def getReadableTime(seconds: float) -> str:
    time_amounts = _getDHMS(seconds)
    first_i = _getFirstSignificantIndex(time_amounts)
    
    time_strings = []
    for i in range(first_i, len(time_amounts)):
        time_strings.append(f"{time_amounts[i]:n}" + "dhms"[i])
    
    return " ".join(time_strings)
        

def _getDHMS(seconds):
    days, seconds = _div(seconds, SECONDS_IN_DAY)    
    hours, seconds = _div(seconds, SECONDS_IN_HOUR)
    minutes, seconds = _div(seconds, SECONDS_IN_MINUTE)

    return (days, hours, minutes, seconds)


def _div(x, y):
    return x // y, x % y


def _getFirstSignificantIndex(time_amounts):
    max_index = len(time_amounts) - 1
    for i in range(max_index):
        if time_amounts[i] != 0:
            return i
    return max_index

