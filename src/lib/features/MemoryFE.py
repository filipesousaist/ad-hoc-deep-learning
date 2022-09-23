from random import random, randint
import numpy as np

from src.lib.features.FeatureExtractor import FeatureExtractor


# 0   X position
# 1   Y position
# 2   Orientation
# 3   Ball X
# 4   Ball Y
# 5   Able to Kick
# 6   Goal Center Proximity
# 7   Goal Center Angle
# 8   Goal Opening Angle
# 9   Proximity to Opponent
# T   Teammate's Goal Opening Angle
# T   Proximity from Teammate i to Opponent
# T   Pass Opening Angle
# 3T  X, Y, and Uniform Number of Teammates    --> 2T  X, Y of Teammates
# 3O  X, Y, and Uniform Number of Opponents    --> 2O  X, Y of Opponents
# +1  Last_Action_Success_Possible
# +1  Stamina

class MemoryFE(FeatureExtractor):
    def __init__(self, num_teammates: int, num_opponents: int):
        super().__init__(num_teammates, num_opponents)

        unused_indices = \
            [10 + 3 * num_teammates + 3 * t + 2 for t in range(num_teammates)] + \
            [10 + 6 * num_teammates + 3 * o + 2 for o in range(num_opponents)]

        self._observation_indices = [i for i in range(12 + 6 * num_teammates + 3 * num_opponents)
                                     if i not in unused_indices]

        self._num_features = len(self._observation_indices)
        self._memorized_features = np.zeros(self._num_features)

    def reset(self):
        for i in range(self._num_features):
            r = randint(0, 1) if i in (5, self._num_features - 2) else random()
            self._memorized_features[i] = 2 * r - 1

    def apply(self, observation: np.ndarray):
        sliced_observation = observation[self._observation_indices]

        for i in range(self._num_features):
            if sliced_observation[i] == -2:
                sliced_observation[i] = self._memorized_features[i]
            else:
                self._memorized_features[i] = sliced_observation[i]

        return sliced_observation
