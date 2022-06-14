import numpy as np
from hfo import SHOOT, DRIBBLE, PASS, MOVE, GO_TO_BALL, REORIENT

from src.lib.observations import distanceToGoal


VALIDATION_FEATURES = {
    SHOOT: (5, 1),
    DRIBBLE: (5, 1),
    PASS: (5, 1),
    MOVE: (5, -1),
    GO_TO_BALL: (5, -1),
    REORIENT: (5, -1)
}


def isActionValid(action: int, observation: np.ndarray) -> bool:
    if action not in VALIDATION_FEATURES:
        return True
    feature, value = VALIDATION_FEATURES[action]
    return int(observation[feature]) == value


def selectActionHandCoded(observation: np.ndarray) -> int:
    if int(observation[5]) == 1:
        if distanceToGoal(*observation[3:5]) <= 0.4:
            return SHOOT
        else:
            return DRIBBLE
    return GO_TO_BALL