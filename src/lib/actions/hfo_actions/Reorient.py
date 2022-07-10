from hfo import REORIENT

from src.lib.actions.hfo_actions.HFOAction import HFOAction


class Dribble(HFOAction):
    @property
    def num_args(self) -> int:
        return 0


    @property
    def index(self) -> int:
        return REORIENT


    @property
    def validation_feature(self) -> int:
        return 5


    @property
    def validation_value(self) -> int:
        return -1
