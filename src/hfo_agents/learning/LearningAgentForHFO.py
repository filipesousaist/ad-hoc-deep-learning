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
from src.lib.observations import ableToKick, OFFENSE_UNUMS
from src.lib.actions.Action import Action
from src.lib.actions.hfo_actions.Move import Move
from src.lib.actions.hfo_actions.NoOp import NoOp
from src.lib.actions.hfo_actions.Pass import Pass
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
        self._saved_observation: np.ndarray = np.array(0)
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

        if custom_actions_data["pass_n"]:
            my_unum = self._hfo.getUnum()
            n = 0
            for u in range(len(OFFENSE_UNUMS)):
                if n == self._num_teammates:
                    break
                unum = OFFENSE_UNUMS[u]
                if unum != my_unum:
                    self._actions.append(Pass(unum))
                    n += 1



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
        if self._auto_moving:
            return Move()
        self._action = self._agent.action(self._extractFeatures(self._observation))
        hfo_action = self._actions[self._action]
        return hfo_action if hfo_action.is_valid(self._observation) else NoOp()


    def _updateAutoMove(self) -> None:
        if self._auto_move:
            self._auto_moving = not ableToKick(self._next_observation)


    def _updateObservation(self) -> None:
        if not self._auto_moving:
            self._saved_observation = self._next_observation


    def _extractFeatures(self, observation: np.ndarray) -> np.ndarray:
        return extractFeatures(observation) if self._custom_features else observation


    def _atEpisodeStart(self) -> None:
        self._agent._last_hidden = None

        self._episode_loss = 0
        self._num_timesteps = 0

        self._action = None
        self._saved_observation = None

        self._updateAutoMove()
        self._updateObservation()



    def _atTimestepStart(self) -> None:
        self._info = {}


    def _atTimestepEnd(self) -> None:
        self._updateAutoMove()

        is_terminal = self._status != IN_GAME
        if (not self._auto_moving or is_terminal) and self._saved_observation is not None:
            self._num_timesteps += 1

            features = self._extractFeatures(self._saved_observation)
            next_features = self._extractFeatures(self._next_observation)

            timestep = Timestep(features, self._action, self._reward_function[self._status],
                                next_features, is_terminal, {})

            self._info.update(self._agent.reinforcement(timestep) or {})
            if "Loss" in self._info:
                self._episode_loss += self._info["Loss"]

        self._updateObservation()


    def _atEpisodeEnd(self) -> None:
        print(self._info)


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
