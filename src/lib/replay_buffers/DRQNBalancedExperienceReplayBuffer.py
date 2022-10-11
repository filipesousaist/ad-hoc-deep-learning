from typing import List

from yaaf import Timestep

from src.lib.replay_buffers.BalancedExperienceReplayBuffer import BalancedExperienceReplayBuffer


class DRQNBalancedExperienceReplayBuffer(BalancedExperienceReplayBuffer):
    @staticmethod
    def _reward(item: List[Timestep]) -> int:
        return item[-1].reward
