def readTxt(path):
    with open(path, "r") as file:
        lines = file.readlines()

        keyValuePairs = [line.split(":") for line in lines]
        keyValueDict = {}

        for pair in keyValuePairs:
            for i in range(2):
                pair[i] = pair[i].strip("\n\r\t ")
            keyValueDict[pair[0]] = pair[1]
        
        return keyValueDict


def writeTxt(path, keyValueDict):
    keyValuePairs = list([(key, keyValueDict[key]) for key in keyValueDict.keys()])
    keyValuePairs.sort(key = lambda pair: pair[0])

    with open(path, "w") as file:
        for pair in keyValuePairs:
            file.write(str(pair[0]) + ": " + str(pair[1]) + "\n")