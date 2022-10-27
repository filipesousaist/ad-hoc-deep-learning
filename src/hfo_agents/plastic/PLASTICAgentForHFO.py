import os
import pickle
from abc import abstractmethod
from random import randint
from typing import List, Dict, Union

import numpy as np
from hfo import IN_GAME, GOAL
from yaaf import Timestep
from yaaf.agents.dqn import DQNAgent
from yaaf.policies import action_from_policy

from src.hfo_agents.AgentForHFO import AgentForHFO
from src.hfo_agents.plastic.Knowledge import Knowledge
from src.lib.actions import getValidAction
from src.lib.actions.Action import Action
from src.lib.actions.hfo_actions.NoOp import NoOp
from src.lib.actions.hfo_actions.Pass import Pass
from src.lib.actions.parsing import parseActions, parseCustomActions
from src.lib.agents import loadReplayBuffer
from src.lib.features.FeatureExtractor import FeatureExtractor
from src.lib.features.default import getDefaultFeatures
from src.lib.features.extractors import getFeatureExtractor
from src.lib.observations import OFFENSE_UNUMS
from src.lib.paths import getPath
from src.lib.reward import parseRewardFunction


class PLASTICAgentForHFO(AgentForHFO):
    def __init__(self, directory: str, teams_directory: str, eta: float, port: int = -1, team: str = "",
                 input_loadout: int = 0, setup_hfo: bool = True, multiprocessing: bool = False):
        super().__init__(directory, port, team, input_loadout, setup_hfo, multiprocessing)

        self._teams_directory = teams_directory
        self._eta = eta

        self._actions: List[Action] = []
        self._a: int = -1
        self._num_actions: int = 0
        self._filter_policy: bool = False

        self._reward_function: Dict[Union[int, str], int] = {}
        self._feature_extractors: List[FeatureExtractor] = []

        self._num_features: int = 0
        self._saved_features: np.ndarray = np.array(0)
        self._next_features: np.ndarray = np.array(0)

        self._all_knowledge: List[Knowledge] = []
        self._num_known_teams: int = 0
        self._behavior_distribution: np.ndarray = np.array(0)

        self._first_episode = True

        self._results = {}

        self._storeInputData()


    def _inputPurpose(self) -> str:
        return "plastic_agent"


    def _storeInputData(self) -> None:
        def _isCustomAction(action) -> bool:
            return isinstance(action, str) and action.startswith("_")

        self._actions = parseActions([action for action in self._input_data["actions"] if not _isCustomAction(action)])
        self._storeCustomActions([action for action in self._input_data["actions"] if _isCustomAction(action)])
        self._num_actions = len(self._actions)

        self._handleOptionalArguments()

        self._reward_function = parseRewardFunction(self._input_data["reward_function"])

        observation_size = 12 + 3 * self._num_opponents + 6 * self._num_teammates
        self._num_features = self._extractFeatures(np.zeros(observation_size)).shape[0]

        self._loadKnowledge(self._input_data["known_teams"], self._input_data["agent_parameters"])


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
        if "filter_policy" in self._input_data:
            self._filter_policy = self._input_data["filter_policy"]


    def _loadKnowledge(self, known_teams: List[str], agent_parameters: dict) -> None:
        for team in known_teams:
            agent = self._createAgent(self._num_features, self._num_actions, agent_parameters)
            agent.eval()
            knowledge_path = getPath(os.path.join(self._teams_directory, team), "knowledge")
            print(f"[INFO] {self.__class__.__name__}: Loading knowledge from path '{knowledge_path}'...")
            agent.load(knowledge_path)
            agent._replay_buffer = loadReplayBuffer(knowledge_path) or agent._replay_buffer

            with open(getPath(knowledge_path, "nn-model"), "rb") as file:
                self._all_knowledge.append(Knowledge(agent, pickle.load(file)))
            print(f"[INFO] {self.__class__.__name__}: Done!")

        self._num_known_teams = len(self._all_knowledge)
        print(f"[INFO] {self.__class__.__name__}: Finished loading knowledge from all {self._num_known_teams} teams.")


    @abstractmethod
    def _createAgent(self, num_features: int, num_actions: int, parameters: dict) -> DQNAgent:
        pass


    def _addFeatureExtractors(self):
        self._feature_extractors = []
        input_features = getDefaultFeatures(self._num_teammates, self._num_opponents)
        for d in range(len(self._input_data["feature_extractors"])):
            data = self._input_data["feature_extractors"][d]
            feature_extractor = getFeatureExtractor(data, input_features) if isinstance(data, str) else \
                getFeatureExtractor(data[0], input_features, *data[1:])
            if feature_extractor.first_only and d > 0:
                exit(f"[ERROR] {self.__class__.__name__}: {feature_extractor.__class__.__name__}" 
                     " must be used in the beginning of the sequence")
            self._feature_extractors.append(feature_extractor)
            input_features = feature_extractor.output_features


    def reset(self):
        self._first_episode = True
        self._behavior_distribution = np.array([1 / self._num_known_teams] * self._num_known_teams)
        self._results = {
            "behavior_distribution": [[p for p in self._behavior_distribution]],
            "goals": [-1]
        }

    def _selectAction(self) -> Action:
        last_action = self._actions[self._a]
        if last_action.usages_left > 0 and self._tryToUse(last_action, renew=False):
            return last_action

        features = self._extractFeatures(self._observation)
        self._a = self._getValidAction(features) if self._filter_policy else self._getAction(features)
        action = self._actions[self._a]
        return action if self._tryToUse(action, renew=True) else NoOp()


    def _getValidAction(self, features: np.ndarray) -> int:
        return getValidAction(self._getPolicy(features), self._actions, self._observation, False)


    def _getAction(self, features: np.ndarray) -> int:
        return action_from_policy(self._getPolicy(features), True)


    def _getPolicy(self, features: np.ndarray) -> np.ndarray:
        return self._all_knowledge[action_from_policy(self._behavior_distribution, True)].policy(features)


    def _tryToUse(self, action: Action, renew: bool) -> bool:
        if action.is_valid(self._observation):
            if renew:
                action.renew()
            action.use()
            #print(f"Used {action.name}, {action.usages_left} usages left")
            return True
        #print(f"Cannot use {action.name}: invalid action")
        #if renew:
        #    print("Used NOOP instead.")
        action.deplete()
        return False


    def _updateFeatures(self) -> None:
        if self._canUpdate():
            self._saved_features = self._next_features


    def _canUpdate(self) -> bool:
        return self._actions[self._a].usages_left == 0


    def _extractFeatures(self, observation: np.ndarray) -> np.ndarray:
        #_FE = FeatureExtractor(getDefaultFeatures(self._num_teammates, self._num_opponents))
        #_FE.printOutputFeatures(observation)

        for feature_extractor in self._feature_extractors:
            observation = feature_extractor.apply(observation)
            #feature_extractor.printOutputFeatures(observation)
        return observation


    def _atEpisodeStart(self) -> None:
        for knowledge in self._all_knowledge:
            knowledge.agent._last_hidden = None

        self._a = randint(0, self._num_actions - 1)
        for action in self._actions:
            action.deplete()
        self._saved_features = None

        for feature_extractor in self._feature_extractors:
            feature_extractor.reset()
        self._next_features = self._extractFeatures(self._next_observation)

        self._updateFeatures()


    def _atTimestepEnd(self) -> None:
        self._next_features = self._extractFeatures(self._next_observation)

        is_terminal = self._status != IN_GAME
        if (self._canUpdate() or is_terminal) and self._saved_features is not None:
            #self._feature_extractors[-1].printOutputFeatures(self._next_features)

            timestep = Timestep(self._saved_features, self._a, self._reward_function[self._status],
                                self._next_features, is_terminal, {})

            #if timestep.reward != 0:
            #    print(timestep)

            for knowledge in self._all_knowledge:
                knowledge.agent.reinforcement(timestep)

            if not self._first_episode:
                self._updateBeliefs()

        self._updateFeatures()


    def _atEpisodeEnd(self) -> None:
        if not self._first_episode:
            self._results["behavior_distribution"].append([p for p in self._behavior_distribution])
            self._results["goals"].append(int(self._status == GOAL))
        self._first_episode = False


    def _updateBeliefs(self):
        for i in range(self._num_known_teams):
            loss = self._all_knowledge[i].getLoss(self._saved_features, self._next_features)
            self._behavior_distribution[i] *= 1 - self._eta * loss
        self._behavior_distribution = self._behavior_distribution / np.sum(self._behavior_distribution)


    @property
    def results(self):
        return self._results
