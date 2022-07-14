from yaaf.agents.dqn.DQNAgent import DQNAgent, MLPDQNAgent

from src.hfo_agents.learning.DQNAgentForHFO import DQNAgentForHFO


class MLPDQNAgentForHFO(DQNAgentForHFO):
    def _createDQNAgent(self, num_features: int, num_actions: int, parameters: dict) -> DQNAgent:
        return MLPDQNAgent(num_features, num_actions, **parameters)
