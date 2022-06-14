from src.lib.ATPO_policy import DRQNAgent

from src.hfo_agents.LearningAgentForHFO import LearningAgentForHFO


class DRQNAgentForHFO(LearningAgentForHFO):
    def _createAgent(self, num_features: int, num_actions: int, parameters: dict) -> DRQNAgent:
        return DRQNAgent(num_features, num_actions, **parameters)
