import numpy as np

from src.lib.actions.hfo_actions.NoOp import NoOp


class Invalid(NoOp):
    def is_valid(self, observation: np.ndarray) -> bool:
        return False
