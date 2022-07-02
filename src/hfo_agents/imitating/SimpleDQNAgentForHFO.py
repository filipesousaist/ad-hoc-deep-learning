from hfo import SHOOT, PASS, DRIBBLE, MOVE

from src.hfo_agents.learning.DQNAgentForHFO import DQNAgentForHFO


params = {'SHT_DST': 0.136664020547, 'SHT_ANG': -0.747394386098,
          'PASS_ANG': 0.464086704478, 'DRIB_DST': -0.999052871962}


def _has_better_pos(dist_to_op, goal_angle, pass_angle, curr_goal_angle):
    """Returns True if teammate is in a better attacking position"""
    if pass_angle < params['PASS_ANG']:
        print(f"Pass angle = {pass_angle} < {params['PASS_ANG']}")
    else:
        print(f"PASS ANGLE = {pass_angle} > {params['PASS_ANG']}")

    if curr_goal_angle > goal_angle:
        print("Better goal angle than teammate my_a =", curr_goal_angle, "> tm_a =", goal_angle)
        return False
    if dist_to_op < params['DRIB_DST']:
        print("Teammate too close to opponent: d =", dist_to_op, "<", params['DRIB_DST'])
        return False
    if pass_angle < params['PASS_ANG']:
        print("Small pass angle a =", pass_angle, "<", params['PASS_ANG'])
        return False
    print("Teammate has better position")
    return True


def _can_shoot(goal_dist, goal_angle):
    """Returns True if player may have a good shot at the goal"""
    return bool((goal_dist < params['SHT_DST']) and (goal_angle > params['SHT_ANG']))


class SimpleDQNAgentForHFO(DQNAgentForHFO):
    def _getAction(self):
        if int(self._observation[5]) == 1:
            goal_dist = float(self._observation[6])
            goal_op_angle = float(self._observation[8])
            if _can_shoot(goal_dist, goal_op_angle):
                return SHOOT
            team_list = list(range(self._num_teammates))
            for i in team_list:
                # teammate_uniform_number = self._observation[10 + 3 * self._num_teammates + 3 * i + 2]
                if _has_better_pos(dist_to_op=float(self._observation[10 + self._num_teammates + i]),
                                   goal_angle=float(self._observation[10 + i]),
                                   pass_angle=float(self._observation[10 + 2 * self._num_teammates + i]),
                                   curr_goal_angle=goal_op_angle):
                    print("Pass")
                    return PASS  # , teammate_uniform_number
            return DRIBBLE
        return MOVE
