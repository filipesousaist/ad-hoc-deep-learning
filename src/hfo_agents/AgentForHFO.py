from abc import abstractmethod
from typing import Optional

import numpy as np

from hfo import HFOEnvironment, HIGH_LEVEL_FEATURE_SET, IN_GAME, SERVER_DOWN, QUIT

from src.lib.evaluation.HFOEnvironmentMultiprocessing import HFOEnvironmentMultiprocessing
from src.lib.actions.Action import Action
from src.lib.paths import getPath
from src.lib.input import readInputData


TEAM_PREFIXES = {
    "base": "base",
    "helios": "HELIOS",
    "aut": "AUT",
    "axiom": "AXIOM",
    "cyrus": "CYRUS",
    "gliders": "GLIDERS",
    "yushan": "YUSHAN"
}


class AgentForHFO:
    def __init__(self, directory: str, port: int = -1, team: str = "", input_loadout: int = 0,
                 setup_hfo: bool = True, multiprocessing: bool = False):
        self._directory: str = directory
        self._port: int = port
        self._team: str = team
        self._input_loadout: int = input_loadout
        self._hfo: Optional[HFOEnvironment] = None
        self._input_data: dict = {}

        self._num_opponents: int = -1
        self._num_teammates: int = -1

        self._observation: np.ndarray = np.array(0)
        self._next_observation: np.ndarray = np.array(0)

        self._status: int = -1

        self._multiprocessing: bool = multiprocessing

        self.readInput()
        if setup_hfo:
            self.setupHFO()


    @staticmethod
    def is_learning_agent():
        return False


    @property
    def status(self) -> int:
        return self._status


    def readInput(self):
        self._input_data = readInputData(getPath(self._directory, "input"),
                                         self._inputPurpose(), self._input_loadout)
        print(f"[INFO] 'AgentForHFO.py' loaded loadout {self._input_loadout}, with the following parameters:")
        print(self._input_data)


    def _inputPurpose(self) -> str:
        return "agent"


    def setTeam(self, team: str):
        self._team = team


    def setLoadout(self, loadout: int):
        self._input_loadout = loadout


    def setupHFO(self):
        # Connect to the server with the specified
        # feature set. See feature sets in hfo.py/hfo.hpp.
        team_info = self._team.split("_")
        actual_team = TEAM_PREFIXES[team_info[0]] + "_" + team_info[1]

        self._hfo = HFOEnvironmentMultiprocessing() if self._multiprocessing else HFOEnvironment()
        self._hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                                  '../HFO/bin/teams/base/config/formations-dt',
                                  self._port, 'localhost', actual_team, False)
        print("[INFO] Custom agent uniform number:", self._hfo.getUnum())

        self._num_opponents = self._hfo.getNumOpponents()
        self._num_teammates = self._hfo.getNumTeammates()


    def deleteHFO(self):
        del self._hfo


    def playEpisode(self) -> bool:
        self._status = IN_GAME
        self._next_observation = self._hfo.getState()

        self._atEpisodeStart()

        self._observation = self._next_observation

        while self._status == IN_GAME:
            self._atTimestepStart()

            self._selectAction().execute(self._hfo)

            self._status = self._hfo.step()
            self._next_observation = self._hfo.getState()

            self._atTimestepEnd()

            self._observation = self._next_observation

        # Quit if the server goes down
        if self._status == SERVER_DOWN:
            self._hfo.act(QUIT)
            return False

        self._atEpisodeEnd()

        return True


    def _atEpisodeStart(self) -> None:
        pass


    def _atTimestepStart(self) -> None:
        pass


    def _atTimestepEnd(self) -> None:
        pass


    def _atEpisodeEnd(self) -> None:
        pass


    @abstractmethod
    def _selectAction(self) -> Action:
        pass
