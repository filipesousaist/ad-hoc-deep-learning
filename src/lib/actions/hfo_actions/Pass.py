from hfo import PASS

from src.lib.actions.Action import Action


class Pass(Action):
    @property
    def num_args(self) -> int:
        return 1


    @property
    def index(self) -> int:
        return PASS


    @property
    def validation_feature(self) -> int:
        return 5


    @property
    def validation_value(self) -> int:
        return 1
