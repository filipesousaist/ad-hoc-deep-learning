from io import TextIOWrapper, DEFAULT_BUFFER_SIZE
import sys
import os
import json

DEV_NULL = open(os.devnull, "w")


def enablePrint() -> None:
    sys.stdout = sys.__stdout__


def disablePrint() -> None:
    sys.stdout = DEV_NULL


class Logger(TextIOWrapper):
    def __init__(self, file, group_id):
        super().__init__(file)
        self.file = file
        self.group_id = group_id
        self.console = None

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


class StdoutLogger(Logger):
    def __init__(self, file, group_id):
        super().__init__(file, group_id)
        self.console = sys.__stdout__
        sys.stdout = self


class StderrLogger(Logger):
    def __init__(self, file, group_id):
        super().__init__(file, group_id)
        self.console = sys.__stderr__
        sys.stderr = self


_loggers: "list[Logger]" = []


def logOutput(log_filename: str, open_mode="w", to_stdout: bool = True, to_stderr: bool = True) -> None:
    if not to_stdout and not to_stderr:
        return
    file = open(log_filename, open_mode)
    if to_stdout:
        _loggers.append(StdoutLogger(file, log_filename))
    if to_stderr:
        _loggers.append(StderrLogger(file, log_filename))


def flushOutput(log_filename: str) -> None:
    for logger in _loggers:
        if logger.group_id == log_filename:
            logger.flush()


def readJSON(path: str) -> dict:
    if not os.path.exists(path):
        sys.exit("[ERROR]: Path \"" + path + "\" not found.")

    with open(path, "r") as file:
        return json.load(file)


def readTxt(path: str) -> dict:
    with open(path, "r") as file:
        lines = file.readlines()

        key_value_pairs = [line.split(":") for line in lines]
        key_value_dict = {}

        for pair in key_value_pairs:
            for i in range(2):
                pair[i] = pair[i].strip("\n\r\t ")
            key_value_dict[pair[0]] = pair[1]

        return key_value_dict


def writeTxt(path: str, key_value_dict: dict) -> None:
    key_value_pairs = list([(key, key_value_dict[key]) for key in key_value_dict.keys()])
    key_value_pairs.sort(key=lambda pair: pair[0])

    with open(path, "w") as file:
        for pair in key_value_pairs:
            file.write(str(pair[0]) + ": " + str(pair[1]) + "\n")


# https://stackoverflow.com/questions/17330139/python-printing-a-dictionary-as-a-horizontal-table-with-headers
def printTable(myDict, colList=None):
    """ Pretty print a list of dictionaries (myDict) as a dynamically sized table.
    If column names (colList) aren't specified, they will show in random order.
    Author: Thierry Husson - Use it as you want but don't blame me.
    """
    if not colList: colList = list(myDict[0].keys() if myDict else [])
    myList = [colList]  # 1st row = header
    for item in myDict: myList.append([str(item[col] if item[col] is not None else '') for col in colList])
    colSize = [max(map(len, col)) for col in zip(*myList)]
    formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
    myList.insert(1, ['-' * i for i in colSize])  # Seperating line
    for item in myList: print(formatStr.format(*item))
