import sys, os

DEV_NULL = open(os.devnull, "w")

def enablePrint():
    sys.stdout = sys.__stdout__

def disablePrint():
    sys.stdout = DEV_NULL