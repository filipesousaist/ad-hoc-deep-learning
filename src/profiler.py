import os, sys
import cProfile, pstats, io
from pstats import SortKey

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("module_path", type=str)
args, _ = parser.parse_known_args()

print(sys.argv)

sys.argv = [sys.argv[0]] + sys.argv[3:]

pr = cProfile.Profile()

exec(f"from {args.module_path} import main")

pr.enable()

exec("main()")

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())