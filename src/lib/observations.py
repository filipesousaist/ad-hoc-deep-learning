import numpy as np
from math import sqrt


GOAL_COORDS = [1, 0]
MAX_DISTANCE = 2 * sqrt(2)


def distanceToGoal(x, y) -> float:
    return sqrt((x - GOAL_COORDS[0])**2 + (y - GOAL_COORDS[1])**2) / MAX_DISTANCE


def distanceToNearestEdge(x: float, y: float) -> float:
    return min(1 - abs(x), 1 - abs(y))


def getTeamUniformNumbers(observation: np.ndarray, num_teammates: int) -> list:
    # 10 + 3T + (3i + 2), i = 0, 1, ...
    uniform_numbers = []
    
    for i in range(num_teammates):
        unum = int(observation[12 + 3 * (num_teammates + i)])
        if unum == -2:
            print(f"Unable to get unum for teammate at index {i}")
            return None
        uniform_numbers.append(unum)
    
    return uniform_numbers