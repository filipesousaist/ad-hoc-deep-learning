from io import TextIOWrapper
import sys
import os
import json
from typing import Tuple, List

import numpy as np

from src.lib.paths import getPath

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


def readScoreRate(directory: str, recursive=False) -> Tuple[np.ndarray, List[np.ndarray]]:
    path = getPath(directory, "test-output")
    if os.path.exists(path):
        file = open(path, "r")
        lines = [line.rstrip("\n%").split(" ") for line in file.readlines() if len(line) > 0 and line[0] == "%"]

        print("Return", path)
        return (
            np.array([int(line[3]) for line in lines]),
            [np.array([float(line[-1]) for line in lines])]
        )
    if not recursive:
        exit("[ERROR] io.py/readScoreRate: Path '" + path + "' not found.")

    return _readScoreRateMultipleFiles(directory)


def _readScoreRateMultipleFiles(directory: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    sub_dirs = getSubDirectories(directory)

    x_list = []
    score_rates = []
    for sub_dir in sub_dirs:
        x, y_list = readScoreRate(sub_dir, True)
        x_list.append(x)
        score_rates += y_list

    min_len = min([len(array) for array in (x_list + score_rates)])

    x_list[0] = x_list[0][:min_len]
    for i in range(1, len(x_list)):
        if not np.array_equal(x_list[i][:min_len], x_list[0]):
            exit("[ERROR] io.py/readScoreRate: X values must be compatible for averaging results. "
                 f"Incompatible files: {getPath(sub_dirs[0], 'test-output')} and {getPath(sub_dirs[i], 'test-output')}")

    for i in range(len(score_rates)):
        score_rates[i] = score_rates[i][:min_len]

    return x_list[0], score_rates


def getSubDirectories(directory: str) -> List[str]:
    sub_dirs = []
    for d in os.listdir(directory):
        path = os.path.join(directory, d)
        if os.path.isdir(path):
            sub_dirs.append(path)
    sub_dirs.sort()
    return sub_dirs


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
