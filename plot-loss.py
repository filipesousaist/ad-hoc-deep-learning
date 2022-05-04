#!/usr/bin/env python
# encoding: utf-8

from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st

INPUT_FILE = "output/drqn-offense-agent-for-1v0-TRAIN-results.txt"
GRANULARITY = 50

file = open(INPUT_FILE, "r")
lines = [line.rstrip("\n").split("\t\t") for line in file.readlines() if len(line) > 0 and line[0].isdigit()]

x = np.array([int(line[0]) for line in lines])
y = np.array([(float(line[1]) if line[1] != "None" else np.nan) for line in lines])

x_2d = x[: x.shape[0] // GRANULARITY * GRANULARITY].reshape((-1, GRANULARITY))
y_2d = y[: y.shape[0] // GRANULARITY * GRANULARITY].reshape((-1, GRANULARITY))

x_max = np.max(x_2d, axis=1)
y_mean = np.nanmean(y_2d, axis=1)
y_std = np.nanstd(y_2d, axis=1)

fig, axes = plt.subplots(2)

fig.canvas.manager.set_window_title('DRQN loss') 
fig.supxlabel("Episodes")
fig.supylabel("Loss")

axes[0].plot(x, y)
axes[1].plot(x_max, y_mean, color=(0, 0, 0.4))
axes[1].fill_between(x_max, y_mean - 1.96 * y_std, y_mean + 1.96 * y_std, color=(0.4, 0.8, 1))
plt.show()
