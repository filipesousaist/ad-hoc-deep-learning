SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 60 * SECONDS_IN_MINUTE
SECONDS_IN_DAY = 24 * SECONDS_IN_HOUR

def getReadableTime(seconds):
    days, seconds = _div(seconds, SECONDS_IN_DAY)
    hours, seconds = _div(seconds, SECONDS_IN_HOUR)
    minutes, seconds = _div(seconds, SECONDS_IN_MINUTE)

    return "{:.0f}d {:.0f}h {:.0f}m {:f}s".format(days, hours, minutes, seconds)

def _div(x, y):
    return x // y, x % y