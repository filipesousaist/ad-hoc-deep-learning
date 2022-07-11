from typing import Dict

from hfo import IN_GAME, GOAL, CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME, SERVER_DOWN


_string_to_status: Dict[str, int] = {
    "IN_GAME": IN_GAME,
    "GOAL": GOAL,
    "CAPTURED_BY_DEFENSE": CAPTURED_BY_DEFENSE,
    "OUT_OF_BOUNDS": OUT_OF_BOUNDS,
    "OUT_OF_TIME": OUT_OF_TIME,
    "SERVER_DOWN": SERVER_DOWN
}


def parseRewardFunction(raw_reward_function: Dict[str, int]) -> Dict[int, int]:
    reward_function = {}

    if "default" in raw_reward_function:
        default_value = raw_reward_function["default"]
        for string in _string_to_status:
            reward_function[_string_to_status[string]] = default_value
        del raw_reward_function["default"]

    for status in raw_reward_function:
        reward_function[_string_to_status[status]] = raw_reward_function[status]

    return reward_function
