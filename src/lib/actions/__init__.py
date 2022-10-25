from typing import List

import numpy as np
from yaaf.policies import action_from_policy

from src.lib.actions.Action import Action


def getValidAction(policy: np.ndarray, actions: List[Action], observation: np.ndarray, is_learning: bool) -> int:
    num_actions = len(actions)
    valid_action_indices = [a for a in range(num_actions) if actions[a].is_valid(observation)]
    filtered_policy = np.array([policy[a] if a in valid_action_indices else 0
                                for a in range(num_actions)])
    p_sum = np.sum(filtered_policy)
    normalized_policy = filtered_policy / p_sum if p_sum > 0 \
        else np.array([1 / len(valid_action_indices) if a in valid_action_indices else 0
                       for a in range(num_actions)])

    return action_from_policy(normalized_policy, not is_learning)
