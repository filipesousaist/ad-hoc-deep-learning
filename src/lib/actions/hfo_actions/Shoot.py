from hfo import SHOOT

from src.lib.actions.hfo_actions.HFOAction import HFOAction


class Shoot(HFOAction):
    @property
    def num_args(self) -> int:
        return 0


    @property
    def index(self) -> int:
        return SHOOT


    @property
    def validation_feature(self) -> int:
        return 5


    @property
    def validation_value(self) -> int:
        return 1
