from typing import Type, List, Dict
import re

from src.lib.actions.Action import Action
from src.lib.actions.hfo_actions.Dribble import Dribble
from src.lib.actions.hfo_actions.Move import Move
from src.lib.actions.hfo_actions.Shoot import Shoot
from src.lib.actions.hfo_actions.Pass import Pass
from src.lib.actions.hfo_actions.NoOp import NoOp
from src.lib.actions.hfo_actions.GoToBall import GoToBall
from src.lib.actions.hfo_actions.Reorient import Reorient


_string_to_action: Dict[str, Type[Action]] = {
    "DASH": None,
    "TURN": None,
    "TACKLE": None,
    "KICK": None,
    "KICK_TO": None,
    "MOVE_TO": None,
    "DRIBBLE_TO": None,
    "INTERCEPT": None,
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
    "REORIENT": Reorient
}


def parseActions(actions: List[str]) -> List[Action]:
    parsed_actions = [list(filter(
            lambda string: string != "",
            re.split('[(),]', action.strip(" "))
        )) for action in actions]
    return [_string_to_action[action[0]](*action[1:]) for action in parsed_actions]


def parseCustomActions(actions: List[str]) -> dict:
    data = {
        "auto_move": "_AUTO_MOVE" in actions
    }

    return data
