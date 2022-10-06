from typing import List

import numpy as np

from src.lib.features import F_FLOAT, E_BALL, F_BOOL
from src.lib.features.FeatureExtractor import FeatureExtractor
from src.lib.features.Feature import Feature, find, findRequiredInputFeatures

from src.lib.observations import distanceToNearestEdge


# 0 Able to Kick
# 1 Goal center proximity
# 2 Goal center angle
# 3 Distance from agent to nearest edge
# 4 Distance from ball to nearest edge


class BasicFE(FeatureExtractor):
    def __init__(self, input_features: List[Feature]):
        self._agent_coords_indices = findRequiredInputFeatures(
            input_features,
            [Feature("x", F_FLOAT), Feature("y", F_FLOAT)],
            "BasicFE"
        )
        self._ball_coords_indices = findRequiredInputFeatures(
            input_features,
            [Feature("x", F_FLOAT, E_BALL), Feature("y", F_FLOAT, E_BALL)],
            "BasicFE"
        )

        super(BasicFE, self).__init__(input_features)

    def _createObservationIndices(self) -> List[int]:
        return findRequiredInputFeatures(
            self.input_features,
            [
                Feature("able_to_kick", F_BOOL),
                Feature("goal_center_proximity", F_FLOAT),
                Feature("goal_center_angle", F_FLOAT)
            ],
            "BasicFE"
        ) + [-2, -1]


    def _createOutputFeatures(self) -> List[Feature]:
        return super()._createOutputFeatures() + [
            Feature("distance_to_nearest_edge", F_FLOAT),
            Feature("distance_to_nearest_edge", F_FLOAT, E_BALL)
        ]


    def _modify(self, observation: np.ndarray) -> np.ndarray:
        # New features
        return observation + [
            distanceToNearestEdge(*self._agent_coords_indices),
            distanceToNearestEdge(*self._ball_coords_indices)
        ]
