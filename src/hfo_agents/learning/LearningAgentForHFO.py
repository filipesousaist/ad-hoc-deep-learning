from abc import abstractmethod
from typing import List, Dict, Union

import numpy as np

from hfo import IN_GAME

from yaaf.agents.Agent import Agent
from yaaf import Timestep

from src.hfo_agents.AgentForHFO import AgentForHFO
from src.lib.io import readTxt
from src.lib.paths import getPath
from src.lib.features import extractFeatures
from src.lib.observations import ableToKick
from src.lib.actions.Action import Action
from src.lib.actions.hfo_actions.Move import Move
from src.lib.actions.hfo_actions.NoOp import NoOp
from src.lib.actions.parsing import parseActions, parseCustomActions
from src.lib.reward import parseRewardFunction


class LearningAgentForHFO(AgentForHFO):
    def __init__(self, directory: str, port: int = -1, team: str = "", input_loadout: int = 0, setup_hfo: bool = True,
                 load_parameters: bool = False):
        super().__init__(directory, port, team, input_loadout, setup_hfo)

        self._actions: List[Action] = []
        self._auto_move: bool = False
        self._auto_moving: bool = False
        self._reward_function: Dict[Union[int, str], int] = {}
        self._custom_features: bool = False

        self._action: int = -1
        self._features: np.ndarray = np.array(0)
        self._next_features: np.ndarray = np.array(0)
        self._info: dict = {}

        self._storeInputData(load_parameters)


    @staticmethod
    def is_learning_agent():
        return True


    @abstractmethod
    def _createAgent(self, num_features: int, num_actions: int, parameters: dict) -> Agent:
        pass


    def _storeInputData(self, load_parameters: bool) -> None:
        self._actions = parseActions([action for action in self._input_data["actions"] if not action.startswith("_")])
        self._storeCustomActions([action for action in self._input_data["actions"] if action.startswith("_")])

        self._custom_features = self._input_data["custom_features"]
        self._reward_function = parseRewardFunction(self._input_data["reward_function"])

        num_features = 12 + 3 * self._num_opponents + 6 * self._num_teammates
        if self._custom_features:
            num_features = len(extractFeatures(np.zeros(num_features)))

        agent_parameters = self._input_data["agent_parameters"]
        if load_parameters:
            save_data = readTxt(getPath(self._directory, "save"))
            self._loadParameters(save_data, agent_parameters)

        self._agent: Agent = self._createAgent(num_features, len(self._actions), agent_parameters)


    def _storeCustomActions(self, actions: List[str]) -> None:
        custom_actions_data = parseCustomActions(actions)
        self._auto_move = custom_actions_data["auto_move"]


    def _inputPurpose(self) -> str:
        return "learning_agent"


    def _loadParameters(self, save_data: dict, target_dict: dict) -> None:
        pass


    def saveParameters(self, save_data: dict) -> None:
        pass


    @property
    def average_loss(self) -> float:
        return self._episode_loss / self._num_timesteps if self._num_timesteps > 0 else None


    def _selectAction(self) -> Action:
        latest_observation = self._next_observation if self._auto_moving else self._observation
        if self._auto_move:
            self._auto_moving = not ableToKick(latest_observation)
            if self._auto_moving:
                return Move()
        self._action = self._agent.action(self._features)
        hfo_action = self._actions[self._action]
        return hfo_action if hfo_action.is_valid(latest_observation) else NoOp()


    def _extractFeatures(self, observation: np.ndarray) -> np.ndarray:
        return extractFeatures(observation) if self._custom_features else observation


    def _atEpisodeStart(self) -> None:
        self._agent._last_hidden = None

        self._episode_loss = 0
        self._num_timesteps = 0

        self._auto_moving = False

        self._features = self._extractFeatures(self._observation)
        self._action = None


    def _atTimestepStart(self) -> None:
        self._info = {}


    def _atTimestepEnd(self) -> None:
        is_terminal = self._status != IN_GAME
        if (not self._auto_moving or is_terminal) and self._action is not None:
            self._num_timesteps += 1

            self._next_features = self._extractFeatures(self._next_observation)

            timestep = Timestep(self._features, self._action, self._reward_function[self._status],
                                self._next_features, is_terminal, {})

            self._features = self._next_features

            self._info.update(self._agent.reinforcement(timestep) or {})
            if "Loss" in self._info:
                self._episode_loss += self._info["Loss"]
        print(self._info)


    def _updateObservation(self) -> None:
        if not self._auto_moving:
            super()._updateObservation()


    def _atEpisodeEnd(self) -> None:
        pass #print(self._info)


    def setLearning(self, value) -> None:
        if value:
            self._agent.train()
        else:
            self._agent.eval()


    @property
    def is_learning(self) -> bool:
        return self._agent.trainable


    @property
    def agent(self):
        return self._agent


    @property
    def exploration_rate(self) -> float:
        return -1.0


    @property
    def total_training_timesteps(self) -> int:
        return self._agent.total_training_timesteps


    def save(self, directory: str) -> None:
        self._agent.save(directory)


    def load(self, directory: str) -> None:
        self._agent.load(directory)
