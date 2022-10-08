from random import random

from hfo import SHOOT, PASS, DRIBBLE, MOVE

from yaaf.agents.Agent import Agent

from src.lib.actions.Action import Action
from src.hfo_agents.learning.MLPDQNAgentForHFO import MLPDQNAgentForHFO


class ImitatingDQNAgentForHFO(MLPDQNAgentForHFO):
    def _createAgent(self, num_features: int, num_actions: int, parameters: dict) -> Agent:
        self._initial_chance_to_imitate: float = parameters["initial_chance_to_imitate"]
        self._final_chance_to_imitate: float = parameters["final_chance_to_imitate"]
        self._steps_to_imitate: int = parameters["steps_to_imitate"]
        self._steps_imitating: int = 0

        actions_set = {SHOOT, PASS, DRIBBLE, MOVE}
        if set(self._actions) != actions_set:
            exit(f"[Error] BaseDQNAgentForHFO should have the actions {actions_set}, but got {self._actions}.")
        if self._auto_move:
            exit("[Error] Auto move is not available for BaseDQNAgentForHFO.")

        del parameters["initial_chance_to_imitate"], \
            parameters["final_chance_to_imitate"], \
            parameters["steps_to_imitate"]
        return super()._createAgent(num_features, num_actions, parameters)


    def _inputPurpose(self) -> str:
        return "imitating_agent"


    def _chance_to_imitate(self) -> float:
        steps_fraction = min(self._steps_imitating / self._steps_to_imitate, 1)
        return self._initial_chance_to_imitate * (1 - steps_fraction) + \
               self._final_chance_to_imitate * steps_fraction


    def _getAction(self) -> Action:
        pass


    def _selectAction(self) -> Action:
        self._info["chance_to_imitate"] = self._chance_to_imitate()
        if self.is_learning and random() < self._info["chance_to_follow_base"]:
            hfo_action = self._getAction()
            self._a = self._actions.index(hfo_action)
            return hfo_action
        return super()._selectAction()
