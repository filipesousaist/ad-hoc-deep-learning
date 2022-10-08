from src.lib.actions.Action import Action


class RepeatedAction(Action):
    def __init__(self, action: Action, duration: int, *args):
        self._action = action

        super().__init__(*args)

        self._duration = duration


    @property
    def index(self) -> int:
        return self._action.index


    @property
    def name(self) -> str:
        return "Repeated_" + super().name


    @property
    def num_args(self) -> int:
        return self._action.num_args


    @property
    def validation_feature(self) -> int:
        return self._action.validation_feature


    @property
    def validation_value(self) -> int:
        return self._action.validation_value


    def renew(self) -> None:
        self._usages_left = self._duration
