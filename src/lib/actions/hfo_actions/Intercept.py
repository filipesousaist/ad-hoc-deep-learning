from hfo import INTERCEPT

from src.lib.actions.Action import Action


class Intercept(Action):
    @property
    def num_args(self) -> int:
        return 0


    @property
    def index(self) -> int:
        return INTERCEPT


    @property
    def validation_feature(self) -> int:
        return 5


    @property
    def validation_value(self) -> int:
        return -1
