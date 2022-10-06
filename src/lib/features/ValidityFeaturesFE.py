from typing import List

import numpy as np

from src.lib.features import F_BOOL, E_TEAMMATE, E_OPPONENT, E_BALL
from src.lib.features.FeatureExtractor import FeatureExtractor
from src.lib.features.Feature import Feature
from src.lib.features.Entity import Entity, getAllEntities


class ValidityFeaturesFE(FeatureExtractor):
    def __init__(self, input_features: List[Feature]):
        self._entities: List[Entity] = getAllEntities(input_features, [E_TEAMMATE, E_OPPONENT, E_BALL])

        super(ValidityFeaturesFE, self).__init__(input_features)


    def _createObservationIndices(self) -> List[int]:
        return list(range(self.num_input_features + len(self._entities)))


    def _createOutputFeatures(self):
        return self.input_features + \
               [Feature("valid", F_BOOL, entity.type, entity.index)
                for entity in self._entities]


    def _modify(self, observation: np.ndarray) -> np.ndarray:
        valid_features = [1 for _ in self._entities]

        for entity in self._entities:
            for f in entity.feature_indices:
                if observation[f] == -2:
                    valid_features[entity.list_index] = -1

        return np.concatenate((observation, np.array(valid_features)))
