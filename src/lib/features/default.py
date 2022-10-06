from typing import List

from src.lib.features import F_FLOAT, F_BOOL, E_TEAMMATE, E_OPPONENT, E_BALL
from src.lib.features.Feature import Feature


def getDefaultFeatures(num_teammates: int, num_opponents: int) -> List[Feature]:
    default_features = [
        Feature("x",                        F_FLOAT),
        Feature("y",                        F_FLOAT),
        Feature("orientation",              F_FLOAT),
        Feature("x",                        F_FLOAT, E_BALL),
        Feature("y",                        F_FLOAT, E_BALL),
        Feature("able_to_kick",             F_BOOL),
        Feature("goal_center_proximity",    F_FLOAT),
        Feature("goal_center_angle",        F_FLOAT),
        Feature("goal_opening_angle",       F_FLOAT),
        Feature("proximity_to_opponent",    F_FLOAT)
    ]

    for feature_name in ("goal_opening_angle", "proximity_to_opponent", "pass_opening_angle"):
        for t in range(1, num_teammates + 1):
            default_features.append(Feature(feature_name, F_FLOAT, E_TEAMMATE, t))

    for t in range(1, num_teammates + 1):
        default_features.extend([
            Feature(f"x",              F_FLOAT, E_TEAMMATE, t),
            Feature(f"y",              F_FLOAT, E_TEAMMATE, t),
            Feature(f"uniform_number", F_FLOAT, E_TEAMMATE, t)
        ])

    for o in range(1, num_opponents + 1):
        default_features.extend([
            Feature(f"x",              F_FLOAT, E_OPPONENT, o),
            Feature(f"y",              F_FLOAT, E_OPPONENT, o),
            Feature(f"uniform_number", F_FLOAT, E_OPPONENT, o)
        ])

    default_features.extend([
        Feature("last_action_success_possible", F_BOOL),
        Feature("stamina",                      F_FLOAT)
    ])

    return default_features
