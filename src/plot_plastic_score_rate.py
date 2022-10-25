#!/usr/bin/env python
# encoding: utf-8
import os.path
from typing import Tuple, List, Type

from matplotlib import pyplot as plt
import numpy as np

import argparse

from src.lib.io import readScoreRate, getSubDirectories, readJSON
from src.lib.paths import getPath, DEFAULT_DIRECTORY

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
    parser.add_argument("-c", "--confidence-intervals", action="store_true")
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

    l_w = args.left_width or DEFAULT_GRAPH_WIDTH
    r_w = args.right_width or DEFAULT_LEGEND_WIDTH

    episodes = []
    score_rates = []
    stds = []
    numbers_of_trials = []

    will_plot = [True] * num_directories

    for d in range(num_directories):
        results_path = getPath(directories[d], "plastic-results")
        if not os.path.exists(results_path):
            print(f"Skipping '{directories[d]}': No data.")
            continue

        results = readJSON(results_path)
        num_trials = len(results)
        numbers_of_trials.append(num_trials)
        if num_trials == 0:
            will_plot[d] = False
            for l in (episodes, score_rates, stds):
                l.append(np.array([]))
            print(f"Skipping '{directories[d]}': No data.")
            continue

        num_episodes_per_trial = len(list(results.values())[0]["goals"]) - 1
        episodes.append(np.arange(1, num_episodes_per_trial + 1))

        score_rates.append(np.zeros((num_episodes_per_trial,)))

        for trial in results.values():
            score_rates[d] += np.array(trial["goals"][1:])

        stds.append(np.array([
            np.std(np.array([1] * int(score_rates[d][i]) + [0] * (num_trials - int(score_rates[d][i]))))
            for i in range(num_episodes_per_trial)
        ]))

        score_rates[d] /= num_trials

    fig, ax_dict = plt.subplot_mosaic(
        [["top"] * l_w + ["BLANK"] * r_w] * 4,
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

            plot(ax_dict["top"], episodes[d], score_rates[d], stds[d], numbers_of_trials[d],
                 LINE_COLORS[d], AREA_COLORS[d], args.confidence_intervals, label or directories[d])

    ax_dict["top"].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_dict["top"].set_xlabel(f"Episode")

    plt.show()


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
