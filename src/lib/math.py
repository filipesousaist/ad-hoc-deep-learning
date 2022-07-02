import numpy as np


def get_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # Check for invalid values:
    for array in [a, b, c]:
        if array[0] == -2 or array[1] == -2:
            return 0.0

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return float(np.degrees(angle))
