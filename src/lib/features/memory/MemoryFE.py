import numpy as np

from src.lib.features.FeatureExtractor import FeatureExtractor


class MemoryFE(FeatureExtractor):
    def __init__(self, input_features: list):
        super().__init__(input_features)

        self._memorized_features = np.zeros(self.num_input_features)


    def reset(self):
        for i in range(self.num_input_features):
            self._memorized_features[i] = -2


    def _modify(self, observation: np.ndarray):
        for i in range(self.num_input_features):
            if observation[i] == -2:
                observation[i] = self._memorized_features[i]
            else:
                self._memorized_features[i] = observation[i]

        return observation
