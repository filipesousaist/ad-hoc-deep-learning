from yaaf.agents.dqn import MLPDQNAgent

from src.hfo_agents.learning.LearningAgentForHFO import LearningAgentForHFO


class DQNAgentForHFO(LearningAgentForHFO):
    def _createAgent(self, num_features: int , num_actions: int, parameters: dict) -> MLPDQNAgent:
        return MLPDQNAgent(num_features, num_actions, **parameters)