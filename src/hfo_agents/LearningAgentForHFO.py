from abc import abstractmethod
from typing import Type
import numpy as np

from hfo import NOOP
# Actions for globals()
from hfo import SHOOT, DRIBBLE, PASS, MOVE, GO_TO_BALL, REORIENT
# Statuses for globals()
from hfo import IN_GAME, GOAL, CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME

from yaaf.agents import Agent
from yaaf import Timestep

from src.hfo_agents.AgentForHFO import AgentForHFO
from src.lib.actions import isActionValid
from src.lib.features import extractFeatures

class LearningAgentForHFO(AgentForHFO):
    def __init__(self, directory: str, port: int, team: str, input_loadout: int = 0):
        super().__init__(directory, port, team, input_loadout)

        self._agent: Type[Agent]
        self._episode_loss: int
        self._num_timesteps: int 

        self._actions: "list[int]"
        self._reward_function: dict
        self._custom_features: bool
        
        self._action: int
        self._features: np.ndarray
        self._next_features: np.ndarray
        self._info: dict

        self._storeInputData()


    @abstractmethod
    def _createAgent(self, num_features: int, num_actions: int, parameters: dict) -> Type[Agent]:
        pass


    def _storeInputData(self) -> None:
        self._input_data["HFO_actions"] = \
            [globals()[action] for action in self._input_data["actions"]]
        self._actions = self._input_data["HFO_actions"]

        self._custom_features = self._input_data["custom_features"]

        self._reward_function = {}
        for status in self._input_data["reward_function"]:
            index = status if status == "default" else globals()[status]
            self._reward_function[index] = self._input_data["reward_function"][status]

        num_features = 12 + 3 * self._num_opponents + 6 * self._num_teammates
        if self._custom_features:
            num_features = len(extractFeatures(np.zeros(num_features)))
        
        num_actions = len(self._actions)
        
        self._agent = self._createAgent(num_features, num_actions, self._input_data["agent_parameters"])


    def _inputPurpose(self) -> str:
        return "learning_agent"


    @property
    def average_loss(self) -> float:
        return self._episode_loss / self._num_timesteps if self._num_timesteps > 0 else None


    def _selectAction(self) -> int:
        self._action = self._agent.action(self._features)
        hfo_action = self._actions[self._action]
        return hfo_action if isActionValid(hfo_action, self._observation) else NOOP


    def _extractFeatures(self, observation):
        return extractFeatures(observation) if self._custom_features else observation
    
    
    def _atEpisodeStart(self) -> None:
        self._agent._last_hidden = None

        self._episode_loss = 0
        self._num_timesteps = 0    
        self._info = {}

        self._features = self._extractFeatures(self._observation)


    def _atEpisodeEnd(self) -> None:
        print(self._info)

    
    def _atTimestepStart(self) -> None:
        self._num_timesteps += 1


    def _atTimestepEnd(self) -> None:
        self._next_features = self._extractFeatures(self._next_observation)

        timestep = Timestep(self._features, self._action, self._reward(self._status), 
            self._next_features, self._status != IN_GAME, {})

        self._features = self._next_features

        self._info = self._agent.reinforcement(timestep)
        if self._info != None and "Loss" in self._info:
            self._episode_loss += self._info["Loss"]


    def _reward(self, status):
        return self._reward_function[status] if status in self._reward_function else \
               self._reward_function["default"]
    

    def setLearning(self, value):
        if value:
            self._agent.train()
        else:
            self._agent.eval()


    def saveNetwork(self, directory: str):
        self._agent.save(directory)
    

    def loadNetwork(self, directory: str):
        self._agent.load(directory)
