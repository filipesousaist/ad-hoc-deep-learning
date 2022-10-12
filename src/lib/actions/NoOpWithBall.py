from src.lib.actions.hfo_actions.NoOp import NoOp


class NoOpWithBall(NoOp):
    @property
    def validation_feature(self) -> int:
        return 5

    @property
    def validation_value(self) -> int:
        return 1
