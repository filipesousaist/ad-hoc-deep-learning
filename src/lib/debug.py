import numpy as np

from src.lib.io import printTable


def printObservationsAndMemoryFeatures(observation: np.ndarray, features: np.ndarray, num_teammates: int,
                                       num_opponents: int) -> None:
    dict_list = []
    dict_list.extend([
        {"Indices": [0, 1, 2], "Names": "X, Y, Orientation", "Observations": observation[0:3],
         "Features": features[0:3]},
        {"Indices": [3, 4], "Names": "Ball X, Ball Y", "Observations": observation[3:5], "Features": features[3:5]},
        {"Indices": [5], "Names": "Able To Kick", "Observations": observation[5:6], "Features": features[5:6]},
        {"Indices": [6, 7], "Names": "Goal Center Proximity, Goal Center Angle", "Observations": observation[6:8],
         "Features": features[6:8]},
        {"Indices": [8, 9], "Names": "Goal Opening Angle, Proximity to Opponent", "Observations": observation[8:10],
         "Features": features[8:10]}
    ])
    for t in range(num_teammates):
        i = 10 + t
        dict_list.append({"Indices": [i], "Names": f"Teammate {t + 1} Goal Opening Angle",
                          "Observations": observation[i:i + 1], "Features": features[i:i + 1]})
    for t in range(num_teammates):
        i = 10 + num_teammates + t
        dict_list.append({"Indices": [i], "Names": f"Teammate {t + 1} Proximity to Opponent",
                          "Observations": observation[i:i + 1], "Features": features[i:i + 1]})
    for t in range(num_teammates):
        i = 10 + 2 * num_teammates + t
        dict_list.append({"Indices": [i], "Names": f"Teammate {t + 1} Pass Opening Angle",
                          "Observations": observation[i:i + 1], "Features": features[i:i + 1]})
    for t in range(num_teammates):
        i = 10 + 3 * num_teammates + 3 * t
        j = 10 + 3 * num_teammates + 2 * t
        dict_list.append({"Indices": [i, i + 1, i + 2], "Names": f"Teammate {t + 1} X, Y, Unum",
                          "Observations": observation[i:i + 3], "Features": np.concatenate((features[j:j + 2], np.array([np.nan])))})
    for o in range(num_opponents):
        i = 10 + 6 * num_teammates + 3 * o
        j = 10 + 5 * num_teammates + 2 * o
        dict_list.append({"Indices": [i, i + 1, i + 2], "Names": f"Opponent {o + 1} X, Y, Unum",
                          "Observations": observation[i:i + 3], "Features": np.concatenate((features[j:j + 2], np.array([np.nan])))})

    i = 10 + 6 * num_teammates + 3 * num_opponents
    j = 10 + 5 * num_teammates + 2 * num_opponents
    dict_list.extend([
        {"Indices": [i], "Names": "Last Action Success Possible",
         "Observations": observation[i:i + 1], "Features": features[j:j + 1]},
        {"Indices": [i + 1], "Names": "Stamina",
         "Observations": observation[i + 1:i + 2], "Features": features[j + 1: j + 2]}
    ])
    printTable(dict_list)
