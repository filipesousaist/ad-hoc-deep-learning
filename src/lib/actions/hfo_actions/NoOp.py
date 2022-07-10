from hfo import NOOP

from src.lib.actions.hfo_actions.HFOAction import HFOAction


class NoOp(HFOAction):
    @property
    def num_args(self) -> int:
        return 0


    @property
    def index(self) -> int:
        return NOOP
