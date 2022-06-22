#!/usr/bin/env python
# encoding: utf-8

from matplotlib import pyplot as plt
import numpy as np

import argparse

from src.lib.paths import getPath, DEFAULT_DIRECTORY

DEFAULT_GRANULARITY = 500

LINE_COLORS = [
    (0.3, 0.1, 0.6),
    (0.3, 0.6, 0.1),
    (0.1, 0.3, 0.6),
    (0.1, 0.6, 0.3),
    (0.6, 0.3, 0.1),
    (0.6, 0.1, 0.3)
]
AREA_COLORS = [tuple([channel + (1 - channel) / 2 for channel in color]) \
    for color in LINE_COLORS]

parser = argparse.ArgumentParser()
parser.add_argument("-D", "--directories", type=str, nargs="+")
parser.add_argument("-c", "--confidence-intervals", action="store_true")
parser.add_argument("-g", "--granularity", type=int)

args = parser.parse_args()

directories = args.directories or [DEFAULT_DIRECTORY]
num_directories = len(directories)
if not (0 < num_directories <= len(LINE_COLORS)):
    exit(f"Number of directories must be between 1 and {len(LINE_COLORS)}") 

granularity = args.granularity or DEFAULT_GRANULARITY

x = []
y = []
x_max = []
y_mean = []
y_std = []

num_train_episodes = None

for d in range(num_directories):
    path = getPath(directories[d], "test-output")
    file = open(path, "r")
    lines = [line.rstrip("\n%").split(" ") for line in file.readlines() if len(line) > 0 and line[0] == "%"]

    x.append(np.array([int(line[3]) for line in lines]))
    y.append(np.array([float(line[-1]) for line in lines]))

    if x[d].shape[0] < 2:
        exit("Need at least 2 points to plot")
    num_train_episodes = x[d][1] - x[d][0]
    
    actual_granularity = (granularity // num_train_episodes) or 1
    granularity = actual_granularity * num_train_episodes

    x_2d = x[d][: x[d].shape[0] // actual_granularity * actual_granularity]\
        .reshape((-1, actual_granularity))
    y_2d = y[d][: y[d].shape[0] // actual_granularity * actual_granularity]\
        .reshape((-1, actual_granularity))

    x_max.append(np.max(x_2d, axis=1))
    y_mean.append(np.nanmean(y_2d, axis=1))
    y_std.append(np.nanstd(y_2d, axis=1))

fig, ax_dict = plt.subplot_mosaic(
        [["top"] * 5 + ["BLANK"]] * 4 + \
        ([["BLANK"] * 6] + [["bottom"] * 5 + ["BLANK"]] * 4 \
         if granularity > num_train_episodes else []),
    empty_sentinel="BLANK")

fig.canvas.manager.set_window_title('Agent score rate') 
fig.supylabel("Score rate (%)")

for d in range(num_directories):
    ax_dict["top"].plot(x[d], y[d], color=LINE_COLORS[d], label=directories[d])
ax_dict["top"].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax_dict["top"].set_xlabel(f"Training Episodes (Granularity = {num_train_episodes})")

if granularity > num_train_episodes:
    for d in range(num_directories):
        ax_dict["bottom"].plot(x_max[d], y_mean[d], color=LINE_COLORS[d])
        if args.confidence_intervals:
            ax_dict["bottom"].fill_between(x_max[d], y_mean[d] - 1.96 * y_std[d], y_mean[d] + 1.96 * y_std[d],
                color=AREA_COLORS[d] + (0.7,))
    ax_dict["bottom"].set_xlabel(f"Training Episodes (Granularity = {granularity})")

plt.show()