from yaaf.agents.dqn import DQNAgent

from src.lib.ATPO_policy import DRQNAgent
from src.hfo_agents.learning.DQNAgentForHFO import DQNAgentForHFO


class DRQNAgentForHFO(DQNAgentForHFO):
    def _createDQNAgent(self, num_features: int, num_actions: int, parameters: dict) -> DQNAgent:
        return DRQNAgent(num_features, num_actions, **parameters)
