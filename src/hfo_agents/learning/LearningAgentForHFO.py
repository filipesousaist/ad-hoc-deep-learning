from abc import abstractmethod
from typing import List, Dict, Union
from random import random, randint
import sys

import numpy as np

from hfo import IN_GAME

from yaaf.agents.Agent import Agent
from yaaf import Timestep

from src.hfo_agents.AgentForHFO import AgentForHFO
from src.lib.io import readTxt
from src.lib.paths import getPath
from src.lib.features.extractors import getFeatureExtractor
from src.lib.features.FeatureExtractor import FeatureExtractor
from src.lib.features.default import getDefaultFeatures
from src.lib.observations import ableToKick, OFFENSE_UNUMS
from src.lib.actions.Action import Action
from src.lib.actions.hfo_actions.Move import Move
from src.lib.actions.hfo_actions.Pass import Pass
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
        self._ignore_auto_move_chance: float = 0.0
        self._see_move_period: int = sys.maxsize
        self._see_move_counter: int = 1

        self._reward_function: Dict[Union[int, str], int] = {}
        self._feature_extractors: List[FeatureExtractor] = []

        self._action: int = -1

        self._saved_features: np.ndarray = np.array(0)
        self._next_features: np.ndarray = np.array(0)

        self._info: dict = {}

        self._storeInputData(load_parameters)

        # Debug
        self._total_timesteps = 0

    @staticmethod
    def is_learning_agent():
        return True

    @abstractmethod
    def _createAgent(self, num_features: int, num_actions: int, parameters: dict) -> Agent:
        pass

    def _storeInputData(self, load_parameters: bool) -> None:
        self._actions = parseActions([action for action in self._input_data["actions"] if not action.startswith("_")])
        self._storeCustomActions([action for action in self._input_data["actions"] if action.startswith("_")])

        self._handleOptionalArguments()

        self._reward_function = parseRewardFunction(self._input_data["reward_function"])

        observation_size = 12 + 3 * self._num_opponents + 6 * self._num_teammates
        num_features = self._extractFeatures(np.zeros(observation_size)).shape[0]

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

    def _handleOptionalArguments(self) -> None:
        if "feature_extractors" in self._input_data:
            self._addFeatureExtractors()
        if "ignore_auto_move_chance" in self._input_data:
            self._ignore_auto_move_chance = self._input_data["ignore_auto_move_chance"]
        if "see_move_period" in self._input_data:
            self._see_move_period = self._input_data["see_move_period"]

    # def _addFeatureExtractorsOld(self):
    #     num_teammates = self._num_teammates
    #     num_opponents = self._num_opponents
    #     for data in self._input_data["feature_extractors"]:
    #         feature_extractor = getFeatureExtractor(data, num_teammates, num_opponents) if isinstance(data, str) else \
    #             getFeatureExtractor(data[0], num_teammates, num_opponents, *data[1:])
    #         self._feature_extractors.append(feature_extractor)
    #         num_teammates = feature_extractor.getOutputNumTeammates()
    #         num_opponents = feature_extractor.getOutputNumOpponents()

    def _addFeatureExtractors(self):
        input_features = getDefaultFeatures(self._num_teammates, self._num_opponents)
        for data in self._input_data["feature_extractors"]:
            feature_extractor = getFeatureExtractor(data, input_features) if isinstance(data, str) else \
                getFeatureExtractor(data[0], input_features, *data[1:])
            self._feature_extractors.append(feature_extractor)
            input_features = feature_extractor.output_features

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
        if self._auto_moving:
            self._see_move_counter = (self._see_move_counter + 1) % self._see_move_period
        if self._auto_move:
            self._auto_moving = not ableToKick(self._next_observation) and random() >= self._ignore_auto_move_chance
        if not self._auto_moving:
            self._see_move_counter = 1

    def _updateFeatures(self) -> None:
        if not self._auto_moving or self._see_move_counter == 0:
            self._saved_features = self._next_features

    def _extractFeatures(self, observation: np.ndarray) -> np.ndarray:
        #_FE = FeatureExtractor(getDefaultFeatures(self._num_teammates, self._num_opponents))
        #_FE.printOutputFeatures(observation)

        for feature_extractor in self._feature_extractors:
            observation = feature_extractor.apply(observation)
            #feature_extractor.printOutputFeatures(observation)
        return observation

    def _atEpisodeStart(self) -> None:
        self._agent._last_hidden = None

        self._episode_loss = 0
        self._num_timesteps = 0

        self._action = randint(0, len(self._actions) - 1)
        self._saved_features = None

        for feature_extractor in self._feature_extractors:
            feature_extractor.reset()
        self._next_features = self._extractFeatures(self._next_observation)

        self._see_move_counter = 1

        self._updateAutoMove()
        self._updateFeatures()

    def _atTimestepStart(self) -> None:
        self._info = {}

    def _atTimestepEnd(self) -> None:
        self._updateAutoMove()
        self._next_features = self._extractFeatures(self._next_observation)

        is_terminal = self._status != IN_GAME
        if (not self._auto_moving or self._see_move_counter == 0 or is_terminal) \
                and self._saved_features is not None:
            self._num_timesteps += 1
            self._total_timesteps += 1

            timestep = Timestep(self._saved_features, self._action, self._reward_function[self._status],
                                self._next_features, is_terminal, {})

            self._info.update(self._agent.reinforcement(timestep) or {})
            if "Loss" in self._info:
                self._episode_loss += self._info["Loss"]

        self._updateFeatures()

    def _atEpisodeEnd(self) -> None:
        print(self._info)
        print("Total timesteps:", self._total_timesteps)

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
