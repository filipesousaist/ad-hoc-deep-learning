from abc import abstractmethod

import numpy as np

from hfo import ACTION_STRINGS, HFOEnvironment


class Action:
    def __init__(self, *args):
        num_args = len(args)
        expected_num_args = self.num_args
        if num_args != expected_num_args:
            exit(f"[ERROR]: Action {self.name} expected {expected_num_args} argument" +
                 f"{'' if expected_num_args == 1 else 's'}, but got {num_args}.")
        self._args = args


    @property
    @abstractmethod
    def name(self) -> str:
        pass


    @property
    @abstractmethod
    def num_args(self) -> int:
        pass


    @property
    def validation_feature(self) -> int:
        return -1


    @property
    def validation_value(self) -> int:
        return 0


    def is_valid(self, observation: np.ndarray) -> bool:
        return self.validation_feature == -1 or \
               int(observation[self.validation_feature]) == self.validation_value


    @abstractmethod
    def execute(self, hfo: HFOEnvironment, observation: np.ndarray) -> None:
        pass
