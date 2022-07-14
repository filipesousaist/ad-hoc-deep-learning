from abc import abstractmethod

from yaaf.agents.Agent import Agent
from yaaf.agents.dqn.DQNAgent import DQNAgent

from src.lib.ATPO_policy import saveReplayBuffer, loadReplayBuffer
from src.hfo_agents.learning.LearningAgentForHFO import LearningAgentForHFO


class DQNAgentForHFO(LearningAgentForHFO):
    def _createAgent(self, num_features: int, num_actions: int, parameters: dict) -> Agent:
        self._dqn_agent = self._createDQNAgent(num_features, num_actions, parameters)
        return self._dqn_agent


    @abstractmethod
    def _createDQNAgent(self, num_features: int, num_actions: int, parameters: dict) -> DQNAgent:
        pass


    def _loadParameters(self, save_data: dict, target_dict: dict):
        super()._loadParameters(save_data, target_dict)
        if "exploration_timesteps_left" in save_data:
            target_dict["final_exploration_step"] = int(save_data["exploration_timesteps_left"])
        if "current_exploration_rate" in save_data:
            target_dict["initial_exploration_rate"] = float(save_data["current_exploration_rate"])


    def saveParameters(self, save_data: dict):
        super().saveParameters(save_data)
        save_data["current_exploration_rate"] = self.exploration_rate
        save_data["exploration_timesteps_left"] = max(self.exploration_timesteps - self.total_training_timesteps, 0)


    @property
    def exploration_rate(self) -> float:
        return self._dqn_agent.exploration_rate


    @property
    def exploration_timesteps(self):
        return self._dqn_agent._exploration_timesteps


    def save(self, directory: str) -> None:
        super().save(directory)
        saveReplayBuffer(self._dqn_agent._replay_buffer, directory)


    def load(self, directory: str) -> None:
        super().load(directory)
        self._agent._replay_buffer = loadReplayBuffer(directory) or self._dqn_agent._replay_buffer