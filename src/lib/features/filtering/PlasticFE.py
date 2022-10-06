from typing import List

from src.lib.features import F_FLOAT, E_TEAMMATE
from src.lib.features.Feature import Feature, findRequiredInputFeatures
from src.lib.features.FeatureExtractor import FeatureExtractor
from src.lib.features.Entity import getAllEntities


# 0   X position
# 1   Y position
# 2   Orientation
# 8   Goal Opening Angle
# 9   Proximity to Opponent
# T   Teammate's Goal Opening Angle
# T   Proximity from Teammate i to Opponent
# T   Pass Opening Angle
# 2T  X, Y of Teammates


class PlasticFE(FeatureExtractor):
    def _createObservationIndices(self) -> List[int]:
        observation_indices = findRequiredInputFeatures(
            self.input_features,
            [
                Feature("x", F_FLOAT),
                Feature("y", F_FLOAT),
                Feature("orientation", F_FLOAT),
                Feature("goal_opening_angle", F_FLOAT),
                Feature("goal_center_angle", F_FLOAT)
            ],
            "PlasticFE"
        )

        teammates = getAllEntities(self.input_features, [E_TEAMMATE])

        # 3T teammate features
        for feature_name in ("goal_opening_angle", "proximity_to_opponent", "pass_opening_angle"):
            for teammate in teammates:
                observation_indices += findRequiredInputFeatures(
                    self.input_features,
                    [Feature(feature_name, F_FLOAT, E_TEAMMATE, teammate.index)],
                    "PlasticFE"
                )

        # X and Y
        for teammate in teammates:
            observation_indices += findRequiredInputFeatures(
                self.input_features,
                [Feature("x", F_FLOAT, E_TEAMMATE, teammate.index),
                 Feature("y", F_FLOAT, E_TEAMMATE, teammate.index)],
                "PlasticFE"
            )

        return observation_indices
