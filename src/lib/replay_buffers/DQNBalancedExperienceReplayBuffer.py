from yaaf import Timestep

from src.lib.replay_buffers.BalancedExperienceReplayBuffer import BalancedExperienceReplayBuffer


class DQNBalancedExperienceReplayBuffer(BalancedExperienceReplayBuffer):
    @staticmethod
    def _reward(item: Timestep) -> int:
        return item.reward
