from abc import abstractmethod

import numpy as np

from hfo import HFOEnvironment

from src.lib.actions.Action import Action
from src.lib.actions.hfo_actions.HFOAction import HFOAction


class SpecialAction(Action):
    @abstractmethod
    def _getHFOAction(self, observation: np.ndarray) -> HFOAction:
        pass


    def execute(self, hfo: HFOEnvironment, observation: np.ndarray) -> None:
        self._getHFOAction(observation).execute()
