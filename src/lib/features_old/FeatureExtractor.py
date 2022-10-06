from abc import abstractmethod

import numpy as np


class FeatureExtractor:
    def __init__(self, num_teammates: int, num_opponents: int):
        self._num_teammates = num_teammates
        self._num_opponents = num_opponents

    @abstractmethod
    def apply(self, observation: np.ndarray):
        raise NotImplementedError()


    def reset(self):
        pass

    def getOutputNumTeammates(self) -> int:
        return self._num_teammates

    def getOutputNumOpponents(self) -> int:
        return self._num_opponents
