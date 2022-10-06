import numpy as np

from src.lib.features.FeatureExtractor import FeatureExtractor


class PartialObservabilityZeroWithValidFeaturesFE(FeatureExtractor):
    def __init__(self, num_teammates: int, num_opponents: int, hide_chance: float):
        super().__init__(num_teammates, num_opponents)
        self._num_features = 12 + 6 * num_teammates + 3 * num_opponents
        self._hide_chance = hide_chance

    def apply(self, observation: np.ndarray):
        should_hide = np.random.random(self._num_features) < self._hide_chance
        return np.array([0 if should_hide[i] else observation[i] for i in range(self._num_features)])
