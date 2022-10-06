import numpy as np

from src.lib.features.FeatureExtractor import FeatureExtractor

from src.lib.observations import distanceToNearestEdge


class BasicFE(FeatureExtractor):
    def apply(self, observation: np.ndarray) -> np.ndarray:
        # New features
        # 0 Able to Kick
        # 1 Goal center proximity
        # 2 Goal center angle
        # 3 Distance from agent to nearest edge
        # 4 Distance from ball to nearest edge

        return np.array([
            *observation[5:8],
            distanceToNearestEdge(*observation[0:2]),
            distanceToNearestEdge(*observation[3:5])
        ])
