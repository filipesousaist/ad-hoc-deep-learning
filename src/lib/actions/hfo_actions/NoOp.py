from hfo import NOOP

from src.lib.actions.Action import Action


class NoOp(Action):
    @property
    def num_args(self) -> int:
        return 0


    @property
    def index(self) -> int:
        return NOOP
