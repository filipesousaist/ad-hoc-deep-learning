from abc import abstractmethod

import numpy as np

from hfo import ACTION_STRINGS, HFOEnvironment

from src.lib.actions.Action import Action


class HFOAction(Action):
    @property
    @abstractmethod
    def index(self) -> int:
        pass


    @property
    def name(self) -> str:
        return ACTION_STRINGS[self.index]


    def execute(self, hfo: HFOEnvironment, observation: np.ndarray) -> None:
        hfo.act(self.index, *self._args)
