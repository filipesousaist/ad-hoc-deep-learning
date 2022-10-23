import pickle

import numpy as np

from src.lib.models.NNModel import NNModel
from src.lib.paths import getPath


class NearestObservationModel(NNModel):
    def __init__(self, nlist: int = 1000, nprobe: int = 1):
        super().__init__(nlist, nprobe)
        self._next_observations = np.array([])


    def fit(self, observations: np.ndarray, next_observations: np.ndarray) -> None:
        super().fit(observations, np.arange(next_observations.shape[0]))
        self._next_observations = next_observations


    def save(self, directory):
        with open(getPath(directory, "nn-model"), "wb") as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load(directory: str) -> "NearestObservationModel":
        with open(getPath(directory, "nn-model"), "rb") as file:
            return pickle.load(file)
