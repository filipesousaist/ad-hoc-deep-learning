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
    parser.add_argument("-D", "--directory", type=str)
    parser.add_argument("-t", "--team", type=str)
    parser.add_argument("-i", "--input-loadout", type=int)
    parser.add_argument("-c", "--confidence-intervals", action="store_true")
    parser.add_argument("-l", "--left-width", type=int,
                        help=f"Width of the graph on the left size of the image. Default: {DEFAULT_GRAPH_WIDTH}")
    parser.add_argument("-r", "--right-width", type=int,
                        help=f"Width of the legend on the right size of the image. Default: {DEFAULT_LEGEND_WIDTH}")
    args = parser.parse_args()

    directory = args.directory or DEFAULT_DIRECTORY
    input_loadout = args.input_loadout or 0

    l_w = args.left_width or DEFAULT_GRAPH_WIDTH
    r_w = args.right_width or DEFAULT_LEGEND_WIDTH

    input_path = getPath(directory, "input")
    known_teams = readJSON(input_path)[str(input_loadout)]["known_teams"]

    num_teams = len(known_teams)

    results_path = getPath(os.path.join(directory, str(input_loadout)), "plastic-results")
    if not os.path.exists(results_path):
        exit("[ERROR]: No data.")
    results_dict = readJSON(results_path)
    results_list = list(results_dict.values())
    filtered_results = list(filter(lambda r: r["correct_team"] == args.team, results_list))
    results = [np.array(filtered_result["behavior_distribution"]) for filtered_result in filtered_results]
    num_episodes_per_trial = results[0].shape[0] - 1

    episodes = np.arange(num_episodes_per_trial + 1)
    num_trials = len(results)
    probabilities = sum(results) / num_trials
    stds = np.std(np.array(results), axis=0)

    fig, ax_dict = plt.subplot_mosaic(
        [["top"] * l_w + ["BLANK"] * r_w] * 4,
        empty_sentinel="BLANK")

    fig.canvas.manager.set_window_title(f'Correct team = {args.team}')
    fig.supylabel("Behavior distribution (%)")

    for t in range(num_teams):
        plot(ax_dict["top"], episodes, probabilities[:, t], stds[:, t], num_trials,
             LINE_COLORS[t], AREA_COLORS[t], args.confidence_intervals, known_teams[t])

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
