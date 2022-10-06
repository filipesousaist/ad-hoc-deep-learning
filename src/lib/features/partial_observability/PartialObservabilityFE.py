import numpy as np

from src.lib.features.FeatureExtractor import FeatureExtractor


class PartialObservabilityFE(FeatureExtractor):
    def __init__(self, input_features: list, hide_chance: float):
        super().__init__(input_features)
        self._hide_chance = hide_chance


    @property
    def hide_chance(self) -> float:
        return self._hide_chance


    def _modify(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
