from hfo import GO_TO_BALL

from src.lib.actions.Action import Action


class GoToBall(Action):
    @property
    def num_args(self) -> int:
        return 0


    @property
    def index(self) -> int:
        return GO_TO_BALL


    @property
    def validation_feature(self) -> int:
        return 5


    @property
    def validation_value(self) -> int:
        return -1
