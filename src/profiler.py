import os
import cProfile, pstats, io
from pstats import SortKey

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("command", type=str)
args = parser.parse_args()

pr = cProfile.Profile()
pr.enable()

os.system(args.command)

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
