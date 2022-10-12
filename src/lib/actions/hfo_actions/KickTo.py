from hfo import KICK_TO

from src.lib.actions.Action import Action


class KickTo(Action):
    @property
    def num_args(self) -> int:
        return 3


    @property
    def index(self) -> int:
        return KICK_TO


    @property
    def validation_feature(self) -> int:
        return 5


    @property
    def validation_value(self) -> int:
        return 1
