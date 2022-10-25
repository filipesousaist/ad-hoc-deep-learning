from yaaf.agents.dqn import DQNAgent

from src.hfo_agents.plastic.PLASTICAgentForHFO import PLASTICAgentForHFO
from src.lib.agents import CustomDRQNAgent


class DRQNPLASTICAgentForHFO(PLASTICAgentForHFO):
    def _createAgent(self, num_features: int, num_actions: int, parameters: dict) -> DQNAgent:
        return CustomDRQNAgent(num_features, num_actions, **parameters)
