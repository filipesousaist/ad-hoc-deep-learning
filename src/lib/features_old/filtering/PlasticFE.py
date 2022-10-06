from src.lib.features.FeatureExtractor import FeatureExtractor

import numpy as np


# 0   X position
# 1   Y position
# 2   Orientation
# 8   Goal Opening Angle
# 9   Proximity to Opponent
# T   Teammate's Goal Opening Angle
# T   Proximity from Teammate i to Opponent
# T   Pass Opening Angle
# 3T  X, Y, and Uniform Number of Teammates    --> 2T  X, Y of Teammates
# 3O  X, Y, and Uniform Number of Opponents    --> 2O  X, Y of Opponents


class PlasticFE(FeatureExtractor):
    def __init__(self, num_teammates: int, num_opponents: int):
        super().__init__(num_teammates, num_opponents)

        self._observation_indices = [0, 1, 2, 8, 9]
        self._observation_indices += [10 + t for t in range(3 * num_teammates)]  # 3T teammate features
        self._observation_indices += [10 + 3 * num_teammates + 3 * t for t in range(num_teammates)]  # X
        self._observation_indices += [10 + 3 * num_teammates + 3 * t + 1 for t in range(num_teammates)]  # Y

    def apply(self, observation: np.ndarray):
        return observation[self._observation_indices]
