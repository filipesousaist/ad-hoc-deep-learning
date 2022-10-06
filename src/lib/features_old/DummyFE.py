import numpy as np

from src.lib.features.FeatureExtractor import FeatureExtractor


# 0   X position
# 1   Y position
# 2   Orientation
# 3   Ball X
# 4   Ball Y
# 5   Able to Kick
# 6   Goal Center Proximity
# 7   Goal Center Angle
# 8   Goal Opening Angle
# 9   Proximity to Opponent
# T   Teammate's Goal Opening Angle            --> T'  Teammate's Goal Opening Angle
# T   Proximity from Teammate i to Opponent    --> T'  Proximity from Teammate i to Opponent
# T   Pass Opening Angle                       --> T'  Proximity from Teammate i to Opponent
# 3T  X, Y, and Uniform Number of Teammates    --> 3T' X, Y, and Uniform Number of Teammates
# 3O  X, Y, and Uniform Number of Opponents    --> 3O' X, Y, and Uniform Number of Opponents
# +1  Last_Action_Success_Possible
# +1  Stamina

class DummyFE(FeatureExtractor):
    def __init__(self, source_num_teammates: int, source_num_opponents: int,
                 target_num_teammates: int, target_num_opponents: int):
        super().__init__(source_num_teammates, source_num_opponents)
        self._target_num_teammates = target_num_teammates
        self._target_num_opponents = target_num_opponents

        self._observation_indices = list(range(10))

        extra_num_teammates = target_num_teammates - source_num_teammates
        if extra_num_teammates < 0:
            exit("[ERROR] DummyFE: target_num_teammates - source_num_teammates must be non-negative.")
        extra_num_opponents = target_num_opponents - source_num_opponents
        if extra_num_opponents < 0:
            exit("[ERROR] DummyFE: target_num_opponents - source_num_opponents must be non-negative.")

        # First 3T teammate features
        for i in range(3):
            self._observation_indices += [10 + i * source_num_teammates + t for t in range(source_num_teammates)] + \
                                         [-1] * (target_num_teammates - source_num_teammates)
        # Second 3T teammate features
        self._observation_indices += [10 + 3 * source_num_teammates + t for t in range(3 * source_num_teammates)] + \
                                     [-1] * (3 * extra_num_teammates)

        # 3O opponent features
        self._observation_indices += [10 + 6 * source_num_teammates + o for o in range(3 * source_num_opponents)] + \
                                     [-1] * (3 * extra_num_opponents)

    def apply(self, observation: np.ndarray):
        dummy_observation = np.concatenate((observation, np.array([0])))
        return dummy_observation[self._observation_indices]

    def getOutputNumTeammates(self) -> int:
        return self._target_num_teammates

    def getOutputNumOpponents(self) -> int:
        return self._target_num_opponents
