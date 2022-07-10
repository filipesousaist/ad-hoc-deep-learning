from abc import abstractmethod

import numpy as np

from hfo import ACTION_STRINGS, HFOEnvironment

from src.lib.actions.Action import Action


class HFOAction(Action):
    @abstractmethod
    @property
    def index(self) -> int:
        pass

    @property
    def name(self) -> str:
        return ACTION_STRINGS[self.index]


    def execute(self, hfo: HFOEnvironment, observation: np.ndarray):
        hfo.act(hfo, self._args)
