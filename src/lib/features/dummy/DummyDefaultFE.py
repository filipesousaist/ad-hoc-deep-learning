from typing import List

from src.lib.features.default import getDefaultFeatures
from src.lib.features.Feature import Feature, find
from src.lib.features.dummy.DummyFE import DummyFE


class DummyDefaultFE(DummyFE):
    def __init__(self, input_features: List[Feature], target_num_teammates: int, target_num_opponents: int):
        self._target_features = getDefaultFeatures(target_num_teammates, target_num_opponents)

        super().__init__(input_features, target_num_teammates, target_num_opponents)


    def first_only(self):
        return True


    def _createObservationIndices(self) -> List[int]:
        observation_indices = []
        for feature in self._target_features:
            found, index = find(self.input_features, feature)
            observation_indices.append(index if found else -1)
        return observation_indices


    def _createOutputFeatures(self) -> List[Feature]:
        return self._target_features
