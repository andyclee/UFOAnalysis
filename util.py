from datetime import datetime
import re

def formatDate(dt):
    #Apply zero padding fro datetime formatting
    dtSplit = dt.split("/", 2)
    dtSplit[2] = dtSplit[2][:4]
    for time in dtSplit:
        if len(time) < 2:
            time = "0" + time

    dtFormat = '/'.join(dtSplit)
    return dtFormat

def dtToWd(dt):
    dtFormatted = formatDate(dt)
    dtObj = datetime.strptime(dtFormatted, '%m/%d/%Y')
    return dtObj.weekday()

def extractDate(dt):
    dtFormatted = formatDate(dt)
    return dtFormatted

def cleanDur(dur):
    if (type(dur) is float):
        return dur

    durSplit = dur.split(".", 2)
    durSplit[0] = re.sub(r'\D', '', durSplit[0])
    if (len(durSplit) > 1):
        durSplit[1] = re.sub(r'\D', '', durSplit[1])
        return '.'.join(durSplit)
    else:
        return re.sub(r'\D', '', dur)

def prepLatCoord(lat):
    return (lat+180)*180.0

def prepLongCoord(longCoord):
    return longCoord+90

def coordSector(rowMaj):
    y = 0
    x = 0
    if (rowMaj % 180 == 0):
        x = 180
        y = rowMaj % 180
    else:
        y = rowMaj // 180
        x = rowMaj - 180 * y

    xSector = x // 45
    ySector = y // 45
    sector = 4 * ySector + xSector

    return sector
