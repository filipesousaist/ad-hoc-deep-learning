from typing import Type, List, Dict, Union

from src.lib.actions.Action import Action

from src.lib.actions.hfo_actions.MoveTo import MoveTo
from src.lib.actions.hfo_actions.Intercept import Intercept
from src.lib.actions.hfo_actions.Move import Move
from src.lib.actions.hfo_actions.Shoot import Shoot
from src.lib.actions.hfo_actions.Pass import Pass
from src.lib.actions.hfo_actions.Dribble import Dribble
from src.lib.actions.hfo_actions.NoOp import NoOp
from src.lib.actions.hfo_actions.GoToBall import GoToBall
from src.lib.actions.hfo_actions.Reorient import Reorient

from src.lib.actions.NoOpWithBall import NoOpWithBall
from src.lib.actions.RepeatedAction import RepeatedAction

_string_to_action: Dict[str, Type[Action]] = {
    "DASH": None,
    "TURN": None,
    "TACKLE": None,
    "KICK": None,
    "KICK_TO": None,
    "MOVE_TO": MoveTo,
    "DRIBBLE_TO": None,
    "INTERCEPT": Intercept,
    "MOVE": Move,
    "SHOOT": Shoot,
    "PASS": Pass,
    "DRIBBLE": Dribble,
    "CATCH": None,
    "NOOP": NoOp,
    "QUIT": None,
    "REDUCE_ANGLE_TO_GOAL": None,
    "MARK_PLAYER": None,
    "DEFEND_GOAL": None,
    "GO_TO_BALL": GoToBall,
    "REORIENT": Reorient,

    "NOOP_WITH_BALL": NoOpWithBall,
    "REPEATED": RepeatedAction
}


ActionArg = List[Union[str, int, "ActionArg"]]


def parseActions(actions: List[Union[str, ActionArg]]) -> List[Action]:
    parsed_actions = []
    for action in actions:
        if isinstance(action, str):
            parsed_actions.append(parseAction(action))
        else:
            parsed_actions.append(parseAction(action[0], *action[1:]))
    return parsed_actions


def parseAction(name: str, *args: Union[str, int, ActionArg]):
    return _string_to_action[name](
        *[
            (parseAction(arg[0], *arg[1:]) if isinstance(arg, list) else
             parseAction(arg) if isinstance(arg, str) else arg)
            for arg in args
        ]
    )


def parseCustomActions(actions: List[str]) -> dict:
    data = {
        "auto_move": "_AUTO_MOVE" in actions,
        "pass_n": "_PASS_N" in actions
    }

    return data
