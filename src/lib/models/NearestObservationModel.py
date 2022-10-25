import pickle

import numpy as np

from src.lib.models.NNModel import NNModel
from src.lib.paths import getPath


def _calculateMaxDistance(observation_size: int) -> float:
    min_array = np.array([-1] * observation_size)
    max_array = np.array([1] * observation_size)
    return np.linalg.norm(max_array - min_array)


class NearestObservationModel(NNModel):
    def __init__(self, nlist: int = 1000, nprobe: int = 1):
        super().__init__(nlist, nprobe)
        self._next_observations: np.ndarray = np.array([])
        self._max_distance: float = 0


    def fit(self, observations: np.ndarray, next_observations: np.ndarray) -> None:
        super().fit(observations, np.arange(next_observations.shape[0]))
        self._next_observations = next_observations
        self._max_distance = _calculateMaxDistance(observations.shape[1])


    def getNextObservation(self, observation: np.ndarray) -> np.ndarray:
        return self._next_observations[self.predict(observation.reshape((1, observation.shape[0])))]


    @property
    def max_distance(self) -> float:
        return self._max_distance



















































































































    def save(self, directory):
        with open(getPath(directory, "nn-model"), "wb") as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load(directory: str) -> "NearestObservationModel":
        with open(getPath(directory, "nn-model"), "rb") as file:
            return pickle.load(file)
