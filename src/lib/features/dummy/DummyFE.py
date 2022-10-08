from typing import List

import numpy as np

from src.lib.features.Feature import Feature
from src.lib.features.FeatureExtractor import FeatureExtractor


class DummyFE(FeatureExtractor):
    def __init__(self, input_features: List[Feature], target_num_teammates: int, target_num_opponents: int):
        self._target_num_teammates = target_num_teammates
        self._target_num_opponents = target_num_opponents

        super().__init__(input_features)


    def _modify(self, observation: np.ndarray) -> np.ndarray:
        dummy_observation = np.concatenate((observation, np.array([-2])))
        return dummy_observation
