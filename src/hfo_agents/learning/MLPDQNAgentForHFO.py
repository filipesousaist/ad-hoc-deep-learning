from typing import List

from yaaf.agents.dqn.DQNAgent import DQNAgent

from src.lib.agents import CustomMLPDQNAgent
from src.hfo_agents.learning.DQNAgentForHFO import DQNAgentForHFO


class MLPDQNAgentForHFO(DQNAgentForHFO):
    def _createDQNAgent(self, num_features: int, num_actions: int, parameters: dict) -> DQNAgent:
        self._mlpdqn_agent = CustomMLPDQNAgent(num_features, num_actions, **parameters)
        return self._mlpdqn_agent


    def _changeableParameters(self) -> List[str]:
        return super(MLPDQNAgentForHFO, self)._changeableParameters() + [
            "initial_episode", "final_episode", "exploration_rate_by_episode"
        ]


    def _changeStaticParameters(self, parameters: dict, parameters_to_change: List[str]) -> None:
        super()._changeStaticParameters(parameters, parameters_to_change)
        if "initial_episode" in parameters_to_change:
            self._mlpdqn_agent._initial_episode = parameters["initial_episode"]
        if "final_episode" in parameters_to_change:
            self._mlpdqn_agent._final_episode = parameters["final_episode"]
        if "exploration_rate_by_episode" in parameters_to_change:
            self._mlpdqn_agent._exploration_rate_by_episode = parameters["exploration_rate_by_episode"]
