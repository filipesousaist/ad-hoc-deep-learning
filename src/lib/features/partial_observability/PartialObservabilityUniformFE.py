import numpy as np

from src.lib.features.partial_observability.PartialObservabilityFE import PartialObservabilityFE


class PartialObservabilityUniformFE(PartialObservabilityFE):
    def _modify(self, observation: np.ndarray) -> np.ndarray:
        should_hide = np.random.random(self.num_input_features) < self.hide_chance
        return np.array([-2 if should_hide[i] else observation[i] for i in range(self.num_input_features)])
