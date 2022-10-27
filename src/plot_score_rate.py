#!/usr/bin/env python
# encoding: utf-8
import os.path
from typing import Tuple, List, Type

from matplotlib import pyplot as plt
import numpy as np

import argparse

from src.lib.io import readScoreRate, getSubDirectories
from src.lib.paths import getPath, DEFAULT_DIRECTORY

DEFAULT_GRANULARITY: int = 500
DEFAULT_GRAPH_WIDTH: int = 4
DEFAULT_LEGEND_WIDTH: int = 3

Color: Type[tuple] = Tuple[float, ...]

LINE_COLORS: List[Color] = [
    (0.66, 0.11, 0.50),
    (0.11, 0.41, 0.21),
    (0.02, 0.33, 0.75),
    (0.35, 0.06, 0.83),
    (0.53, 0.29, 0),
    (0.84, 0.09, 0.17),
    (0.45, 0.64, 0.31)
]

AREA_COLORS: List[Color] = [tuple([channel + (1 - channel) / 3 for channel in color])
                            for color in LINE_COLORS]

AREA_ALPHA: float = 0.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-S", "--super-directories", type=str, nargs="+")
    parser.add_argument("-D", "--directories", type=str, nargs="+")
    parser.add_argument("-O", "--offsets", type=int, nargs="+")
    parser.add_argument("-i", "--ignore-errors", action="store_true")
    parser.add_argument("-c", "--confidence-intervals", action="store_true")
    parser.add_argument("-g", "--granularity", type=int)
    parser.add_argument("-l", "--left-width", type=int,
                        help=f"Width of the graph on the left size of the image. Default: {DEFAULT_GRAPH_WIDTH}")
    parser.add_argument("-r", "--right-width", type=int,
                        help=f"Width of the legend on the right size of the image. Default: {DEFAULT_LEGEND_WIDTH}")
    args = parser.parse_args()

    directories = args.directories or []
    if args.super_directories:
        for super_dir in args.super_directories:
            directories += getSubDirectories(super_dir)
    directories = directories or [DEFAULT_DIRECTORY]
    num_directories = len(directories)
    if not (0 < num_directories <= len(LINE_COLORS)):
        exit(f"[ERROR]: Number of directories ({num_directories}) is outside the valid range "
             f"(between 1 and {len(LINE_COLORS)}).")

    offsets = args.offsets or [0] * num_directories
    num_offsets = len(offsets)
    if num_offsets != num_directories:
        exit(f"[ERROR]: Number of offsets ({num_offsets}) is different from number of directories ({num_directories}).")

    granularity = args.granularity or DEFAULT_GRANULARITY
    l_w = args.left_width or DEFAULT_GRAPH_WIDTH
    r_w = args.right_width or DEFAULT_LEGEND_WIDTH

    n_list = []

    x_list = []
    y_list = []
    y_std_list = []

    x_min_list = []
    y_mean_list = []
    y_mean_std_list = []

    num_train_episodes = None

    will_plot = [True] * num_directories

    for d in range(num_directories):
        x, ys = readScoreRate(directories[d], True, args.ignore_errors)

        num_plots = len(ys)
        n_list.append(num_plots)
        if len(x) == 0 or num_plots == 0:
            will_plot[d] = False
            for l in (x_list, y_list, y_std_list, x_min_list, y_mean_list, y_mean_std_list):
                l.append(np.array([]))
            print(f"Skipping '{directories[d]}': No data.")
            continue

        x += offsets[d]

        x_list.append(x)

        Y = np.array(ys)

        y = np.nanmean(Y, axis=0).reshape((-1,))
        y_list.append(y)
        y_std_list.append(np.nanstd(Y, axis=0).reshape((-1,)))

        if x.shape[0] < 2:
            exit("[ERROR]: Need at least 2 points to plot")
        num_train_episodes = x[1] - x[0]

        actual_granularity = (granularity // num_train_episodes) or 1
        granularity = actual_granularity * num_train_episodes

        x_2d = reshape(x, actual_granularity)
        y_2d = reshape(y, actual_granularity)

        x_min_list.append(np.min(x_2d, axis=1))
        y_mean_list.append(np.nanmean(y_2d, axis=1))
        y_mean_std_list.append(np.nanstd(y_2d, axis=1))

    fig, ax_dict = plt.subplot_mosaic(
        [["top"] * l_w + ["BLANK"] * r_w] * 4 +
        ([["BLANK"] * (l_w + r_w)] + [["bottom"] * l_w + ["BLANK"] * r_w] * 4
         if granularity > num_train_episodes else []),
        empty_sentinel="BLANK")

    fig.canvas.manager.set_window_title('Agent score rate')
    fig.supylabel("Score rate (%)")

    for d in range(num_directories):
        if will_plot[d]:
            label_path = getPath(directories[d], "label")
            label = None
            if os.path.exists(label_path):
                label_file = open(label_path, "r")
                label = label_file.readline()

            plot(ax_dict["top"], x_list[d], y_list[d], y_std_list[d], n_list[d],
                 LINE_COLORS[d], AREA_COLORS[d], args.confidence_intervals, label or directories[d])

    ax_dict["top"].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    granularity_str = f" (Granularity = {num_train_episodes})" if "bottom" in ax_dict else ""
    ax_dict["top"].set_xlabel("Training Episodes" + granularity_str)

    if granularity > num_train_episodes:
        for d in range(num_directories):
            if will_plot[d]:
                plot(ax_dict["bottom"], x_min_list[d], y_mean_list[d], y_mean_std_list[d], n_list[d],
                     LINE_COLORS[d], AREA_COLORS[d], args.confidence_intervals)

        ax_dict["bottom"].set_xlabel(f"Training Episodes (Granularity = {granularity})")

    plt.show()


def reshape(array: np.ndarray, granularity: int) -> np.ndarray:
    return array[: array.shape[0] // granularity * granularity].reshape((-1, granularity))


def plot(ax: plt.Axes, x: np.ndarray, y: np.ndarray, y_std: np.ndarray, n: int,
         color: Color, area_color: Color, use_confidence_intervals: bool, label: str = None) -> None:
    ax.plot(x, y, color=color, label=label)
    if use_confidence_intervals:
        ax.fill_between(x, *confidenceInterval(y, y_std, n), color=area_color + (AREA_ALPHA,))


def confidenceInterval(mean: np.ndarray, std: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    deviation = 1.96 * std / np.sqrt(n)
    return mean - deviation, mean + deviation


if __name__ == "__main__":
    main()
