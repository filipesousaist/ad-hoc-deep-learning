from typing import List

import numpy as np

from src.lib.features.Feature import Feature
from src.lib.io import printTable


class FeatureExtractor:
    def __init__(self, input_features: List[Feature]):
        self._input_features: List[Feature] = input_features
        self._num_input_features: int = len(self._input_features)

        self._observation_indices: List[int] = self._createObservationIndices()

        self._output_features: List[Feature] = self._createOutputFeatures()
        self._num_output_features: int = len(self._output_features)


    def _createObservationIndices(self) -> List[int]:
        return list(range(self._num_input_features))


    def _createOutputFeatures(self) -> List[Feature]:
        return [self._input_features[i] for i in self._observation_indices]


    @property
    def input_features(self) -> List[Feature]:
        return self._input_features[:]


    @property
    def output_features(self) -> List[Feature]:
        return self._output_features[:]


    @property
    def num_input_features(self) -> int:
        return self._num_input_features


    @property
    def num_output_features(self) -> int:
        return self._num_output_features


    def apply(self, observation: np.ndarray) -> np.ndarray:
        modified_observation = self._modify(observation[:])
        return modified_observation[self._observation_indices]


    def _modify(self, observation: np.ndarray) -> np.ndarray:
        return observation


    def reset(self):
        pass


    @property
    def first_only(self):
        return False

    def printOutputFeatures(self, features: np.ndarray):
        header_str = f"*** Output features of {self.__class__.__name__} ***"
        print(header_str)
        printTable([
            {"Index": i, "Feature": self._output_features[i], "Value": features[i]}
            for i in range(self._num_output_features)
        ])
        print("*" * len(header_str))

