from hfo import MOVE_TO

from src.lib.actions.hfo_actions.HFOAction import HFOAction


class MoveTo(HFOAction):
    @property
    def num_args(self) -> int:
        return 2


    @property
    def index(self) -> int:
        return MOVE_TO


    @property
    def validation_feature(self) -> int:
        return 5


    @property
    def validation_value(self) -> int:
        return -1
