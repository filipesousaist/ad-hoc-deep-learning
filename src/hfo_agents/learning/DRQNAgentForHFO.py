from typing import List

from yaaf.agents.dqn import DQNAgent

from src.lib.ATPO_policy import DRQNAgent
from src.hfo_agents.learning.DQNAgentForHFO import DQNAgentForHFO


class DRQNAgentForHFO(DQNAgentForHFO):
    def _createDQNAgent(self, num_features: int, num_actions: int, parameters: dict) -> DQNAgent:
        self._drqn_agent = DRQNAgent(num_features, num_actions, **parameters)
        return self._drqn_agent

    @staticmethod
    def _changeableParameters() -> List[str]:
        return super(DRQNAgentForHFO, DRQNAgentForHFO)._changeableParameters() + \
               ["network_update_frequency", "trajectory_update_length"]

    def _changeStaticParameters(self, parameters: dict, parameters_to_change: List[str]) -> None:
        super()._changeStaticParameters(parameters, parameters_to_change)
        if "network_update_frequency" in parameters_to_change:
            self._dqn_agent._network_update_frequency = parameters["network_update_frequency"]
        if "trajectory_update_length" in parameters_to_change:
            self._drqn_agent.max_sequence_length = parameters["trajectory_update_length"]
