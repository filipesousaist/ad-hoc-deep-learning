from src.lib.actions.Action import Action

from src.lib.actions.hfo_actions.Shoot import Shoot
from src.lib.actions.hfo_actions.Pass import Pass
from src.lib.actions.hfo_actions.Dribble import Dribble
from src.lib.actions.hfo_actions.Move import Move

from src.hfo_agents.AgentForHFO import AgentForHFO


_SHT_DST = 0.136664020547
_SHT_ANG = -0.747394386098
_PASS_ANG = -0.5


def _canShoot(goal_dist, goal_angle):
    """Returns True if if player may have a good shot at the goal"""
    return goal_dist != -2 and goal_angle != -2 and goal_dist < _SHT_DST and goal_angle > _SHT_ANG


def _canPass(pass_angle):
    return pass_angle != -2 and pass_angle > _PASS_ANG


class SimpleAgentForHFO(AgentForHFO):   
    def _selectAction(self) -> Action:
        can_kick = self._observation[5] == 1
        goal_dist = self._observation[6]
        goal_angle = self._observation[8]
        teammate_pass_angle = self._observation[10 + 2 * self._num_teammates]
        teammate_goal_angle = self._observation[10]

        if can_kick:
            if _canShoot(goal_dist, goal_angle):
                return Shoot()
            elif self._num_teammates > 0 and _canPass(teammate_pass_angle) and teammate_goal_angle > goal_angle:
                return Pass(7)
            return Dribble()
        return Move()

    