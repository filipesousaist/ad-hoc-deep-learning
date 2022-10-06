from typing import List

import numpy as np

from src.lib.features import E_TEAMMATE, E_OPPONENT, E_BALL
from src.lib.features.Feature import Feature
from src.lib.features.partial_observability.PartialObservabilityFE import PartialObservabilityFE
from src.lib.features.Entity import getAllEntities


class PartialObservabilityByEntityFE(PartialObservabilityFE):
    def __init__(self, input_features: List[Feature], hide_chance: float):
        super().__init__(input_features, hide_chance)

        self._entities = getAllEntities(self.input_features, [E_TEAMMATE, E_OPPONENT, E_BALL])
        self._num_entities = len(self._entities)


    def _modify(self, observation: np.ndarray) -> np.ndarray:
        should_hide = np.random.random(self._num_entities) < self.hide_chance
        for e in range(self._num_entities):
            if should_hide[e]:
                for f in self._entities[e].feature_indices:
                    observation[f] = -2
        return observation
