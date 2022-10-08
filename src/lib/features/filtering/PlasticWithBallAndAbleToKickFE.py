from typing import List

from src.lib.features import F_BOOL
from src.lib.features.Feature import findRequiredInputFeatures, Feature
from src.lib.features.filtering.PlasticWithBallFE import PlasticWithBallFE


# 0   X position
# 1   Y position
# 2   Orientation
# 3   Ball X
# 4   Ball Y
# 5   Able To Kick
# 8   Goal Opening Angle
# 9   Proximity to Opponent
# T   Teammate's Goal Opening Angle
# T   Proximity from Teammate i to Opponent
# T   Pass Opening Angle
# 2T  X, Y of Teammates


class PlasticWithBallAndAbleToKickFE(PlasticWithBallFE):
    def _createObservationIndices(self) -> List[int]:
        observation_indices = super()._createObservationIndices()
        return observation_indices[0:5] + \
            findRequiredInputFeatures(
                self.input_features,
                [Feature("able_to_kick", F_BOOL)],
                "PlasticWithBallAndAbleToKickFE"
            ) + observation_indices[5:]
