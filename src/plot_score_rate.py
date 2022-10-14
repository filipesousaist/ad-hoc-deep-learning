#!/usr/bin/env python
# encoding: utf-8
import os.path

from matplotlib import pyplot as plt
import numpy as np

import argparse

from src.lib.io import readScoreRate
from src.lib.paths import getPath, DEFAULT_DIRECTORY

DEFAULT_GRANULARITY = 500
DEFAULT_GRAPH_WIDTH = 4
DEFAULT_LEGEND_WIDTH = 3

LINE_COLORS = (
    (0.66, 0.11, 0.50),
    (0.11, 0.41, 0.21),
    (0.02, 0.33, 0.75),
    (0.35, 0.06, 0.83),
    (0.53, 0.29, 0),
    (0.84, 0.09, 0.17),
    (0.45, 0.64, 0.31)
)

AREA_COLORS = [tuple([channel + (1 - channel) / 3 for channel in color])
               for color in LINE_COLORS]

parser = argparse.ArgumentParser()
parser.add_argument("-D", "--directories", type=str, nargs="+")
parser.add_argument("-c", "--confidence-intervals", action="store_true")
parser.add_argument("-g", "--granularity", type=int)
parser.add_argument("-l", "--left-width", type=int,
                    help=f"Width of the graph on the left size of the image. Default: {DEFAULT_GRAPH_WIDTH}")
parser.add_argument("-r", "--right-width", type=int,
                    help=f"Width of the legend on the right size of the image. Default: {DEFAULT_LEGEND_WIDTH}")
args = parser.parse_args()

directories = args.directories or [DEFAULT_DIRECTORY]
num_directories = len(directories)
if not (0 < num_directories <= len(LINE_COLORS)):
    exit(f"Number of directories must be between 1 and {len(LINE_COLORS)}")

granularity = args.granularity or DEFAULT_GRANULARITY
l_w = args.left_width or DEFAULT_GRAPH_WIDTH
r_w = args.right_width or DEFAULT_LEGEND_WIDTH

x = []
y = []
x_max = []
y_mean = []
y_std = []

num_train_episodes = None

for d in range(num_directories):
    data_arrays = readScoreRate(directories[d])
    x.append(data_arrays[0])
    y.append(data_arrays[1])

    if x[d].shape[0] < 2:
        exit("Need at least 2 points to plot")
    num_train_episodes = x[d][1] - x[d][0]

    actual_granularity = (granularity // num_train_episodes) or 1
    granularity = actual_granularity * num_train_episodes

    x_2d = x[d][: x[d].shape[0] // actual_granularity * actual_granularity] \
        .reshape((-1, actual_granularity))
    y_2d = y[d][: y[d].shape[0] // actual_granularity * actual_granularity] \
        .reshape((-1, actual_granularity))

    x_max.append(np.max(x_2d, axis=1))
    y_mean.append(np.nanmean(y_2d, axis=1))
    y_std.append(np.nanstd(y_2d, axis=1))

fig, ax_dict = plt.subplot_mosaic(
    [["top"] * l_w + ["BLANK"] * r_w] * 4 +
    ([["BLANK"] * (l_w + r_w)] + [["bottom"] * l_w + ["BLANK"] * r_w] * 4
     if granularity > num_train_episodes else []),
    empty_sentinel="BLANK")

fig.canvas.manager.set_window_title('Agent score rate')
fig.supylabel("Score rate (%)")

for d in range(num_directories):
    label_path = getPath(directories[d], "label")
    label = None
    if os.path.exists(label_path):
        label_file = open(label_path, "r")
        label = label_file.readline()
    ax_dict["top"].plot(x[d], y[d], color=LINE_COLORS[d], label=(label or directories[d]))

ax_dict["top"].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax_dict["top"].set_xlabel(f"Training Episodes (Granularity = {num_train_episodes})")

if granularity > num_train_episodes:
    for d in range(num_directories):
        ax_dict["bottom"].plot(x_max[d], y_mean[d], color=LINE_COLORS[d])
        if args.confidence_intervals:
            ax_dict["bottom"].fill_between(x_max[d], y_mean[d] - 1.96 * y_std[d], y_mean[d] + 1.96 * y_std[d],
                                           color=AREA_COLORS[d] + (0.3,))
    ax_dict["bottom"].set_xlabel(f"Training Episodes (Granularity = {granularity})")

plt.show()
