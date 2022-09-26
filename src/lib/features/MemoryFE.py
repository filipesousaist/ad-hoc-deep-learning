from random import random, randint
import numpy as np

from src.lib.features.FeatureExtractor import FeatureExtractor

"""
Apply this class before any filtering FEs
"""


class MemoryFE(FeatureExtractor):
    def __init__(self, num_teammates: int, num_opponents: int):
        super().__init__(num_teammates, num_opponents)

        self._num_features = 12 + 6 * num_teammates + 3 * num_opponents
        self._memorized_features = np.zeros(self._num_features)

    def reset(self):
        for i in range(self._num_features):
            r = randint(0, 1) if i in (5, self._num_features - 2) else random()
            self._memorized_features[i] = 2 * r - 1

    def apply(self, observation: np.ndarray):
        sliced_observation = super().apply(observation)

        for i in range(self._num_features):
            if sliced_observation[i] == -2:
                sliced_observation[i] = self._memorized_features[i]
            else:
                self._memorized_features[i] = sliced_observation[i]

        return sliced_observation
