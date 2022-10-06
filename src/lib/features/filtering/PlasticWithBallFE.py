from typing import List

from src.lib.features import F_FLOAT, E_BALL
from src.lib.features.filtering.PlasticFE import PlasticFE
from src.lib.features.Feature import Feature, findRequiredInputFeatures


# 0   X position
# 1   Y position
# 2   Orientation
# 3   Ball X
# 4   Ball Y
# 8   Goal Opening Angle
# 9   Proximity to Opponent
# T   Teammate's Goal Opening Angle
# T   Proximity from Teammate i to Opponent
# T   Pass Opening Angle
# 3T  X, Y, and Uniform Number of Teammates    --> 2T  X, Y of Teammates
# 3O  X, Y, and Uniform Number of Opponents    --> 2O  X, Y of Opponents


class PlasticWithBallFE(PlasticFE):
    def _createObservationIndices(self) -> List[int]:
        observation_indices = super()._createObservationIndices()
        return observation_indices[0:3] + \
            findRequiredInputFeatures(
                self.input_features,
                [Feature("x", F_FLOAT, E_BALL),
                 Feature("y", F_FLOAT, E_BALL)],
                "PlasticWithBallFE"
            ) + observation_indices[3:]
