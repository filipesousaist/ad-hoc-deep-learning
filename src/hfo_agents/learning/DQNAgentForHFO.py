from abc import abstractmethod
from typing import List

import numpy as np
from yaaf import Timestep
from yaaf.agents.Agent import Agent
from yaaf.agents.dqn.DQNAgent import DQNAgent

from src.lib.models.NearestObservationModel import NearestObservationModel
from src.lib.agents import saveReplayBuffer, loadReplayBuffer
from src.hfo_agents.learning.LearningAgentForHFO import LearningAgentForHFO
from src.lib.paths import getPath


def _retrieveLoadedParameters(parameters: dict) -> dict:
    if "_total_training_timesteps" in parameters:
        total_training_timesteps = parameters["_total_training_timesteps"]
        del parameters["_total_training_timesteps"]
        return {"_total_training_timesteps": total_training_timesteps}
    return {}


class DQNAgentForHFO(LearningAgentForHFO):
    def _createAgent(self, num_features: int, num_actions: int, parameters: dict) -> Agent:
        loaded_parameters = _retrieveLoadedParameters(parameters)
        self._dqn_agent = self._createDQNAgent(num_features, num_actions, parameters)
        self._setLoadedParameters(loaded_parameters)
        return self._dqn_agent


    @abstractmethod
    def _createDQNAgent(self, num_features: int, num_actions: int, parameters: dict) -> DQNAgent:
        pass


    def _loadParameters(self, save_data: dict, target_dict: dict) -> None:
        super()._loadParameters(save_data, target_dict)
        if "total_training_timesteps" in save_data:
            target_dict["_total_training_timesteps"] = int(save_data["total_training_timesteps"])
        elif "exploration_timesteps_left" in save_data:
            target_dict["final_exploration_step"] = int(save_data["exploration_timesteps_left"])


    def _setLoadedParameters(self, parameters: dict) -> None:
        if "_total_training_timesteps" in parameters:
            self._dqn_agent._total_training_timesteps.value = parameters["_total_training_timesteps"]


    def saveParameters(self, save_data: dict) -> None:
        super().saveParameters(save_data)
        is_learning = self.is_learning
        self.setLearning(True)

        save_data["current_exploration_rate"] = self.exploration_rate
        save_data["exploration_timesteps_left"] = max(self.exploration_timesteps - self.total_training_timesteps, 0)
        save_data["total_training_timesteps"] = self.total_training_timesteps

        self.setLearning(is_learning)

    def resetParameters(self) -> None:
        self._dqn_agent._total_training_timesteps.value = 0

    @staticmethod
    def _changeableParameters() -> List[str]:
        return [
            "initial_exploration_rate", "final_exploration_rate", "initial_exploration_steps", "final_exploration_step",
            "target_network_update_frequency",
        ]

    def _changeStaticParameters(self, parameters: dict, parameters_to_change: List[str]) -> None:
        if "initial_exploration_rate" in parameters_to_change:
            self._dqn_agent._initial_exploration_rate = parameters["initial_exploration_rate"]
        if "final_exploration_rate" in parameters_to_change:
            self._dqn_agent._final_exploration_rate = parameters["final_exploration_rate"]
        if "initial_exploration_steps" in parameters_to_change:
            self._dqn_agent._replay_start_size = parameters["initial_exploration_steps"]
        if "final_exploration_step" in parameters_to_change:
            self._dqn_agent._exploration_timesteps = parameters["final_exploration_step"]
        if "target_network_update_frequency" in parameters_to_change:
            self._dqn_agent._target_network_update_frequency = parameters["target_network_update_frequency"]

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
        self._dqn_agent._replay_buffer = loadReplayBuffer(directory) or self._dqn_agent._replay_buffer

    def createNNModel(self) -> None:
        timesteps = self._getAllTimesteps()

        observations = np.array([timestep.observation for timestep in timesteps])
        next_observations = np.array([timestep.next_observation for timestep in timesteps])

        model = NearestObservationModel()
        print(f"[INFO] {self.__class__.__name__}: Fitting NNModel to observations...")
        model.fit(observations, next_observations)
        print(f"[INFO] {self.__class__.__name__}: Done!")

        path = getPath(self._directory, "knowledge")
        print(f"[INFO] {self.__class__.__name__}: Saving NNModel to directory '{path}'...")
        model.save(path)
        print(f"[INFO] {self.__class__.__name__}: Done!")


    @abstractmethod
    def _getAllTimesteps(self) -> List[Timestep]:
        pass
