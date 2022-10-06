import numpy as np

from src.lib.features.FeatureExtractor import FeatureExtractor


class InvalidToZeroFE(FeatureExtractor):
    def _modify(self, observation: np.ndarray) -> np.ndarray:
        for i in range(self.num_input_features):
            if observation[i] == -2:
                observation[i] = 0
        return observation
