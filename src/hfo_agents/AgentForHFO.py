from abc import abstractmethod

import numpy as np

from hfo import HFOEnvironment, HIGH_LEVEL_FEATURE_SET, IN_GAME, SERVER_DOWN, PASS, NOOP, QUIT

from src.lib.observations import getTeamUniformNumbers
from src.lib.paths import getPath
from src.lib.input import readInputData


class AgentForHFO:
    def __init__(self, directory: str, port: int = -1, team: str = "", input_loadout: int = 0, setup_hfo: bool = True):
        self._directory: str = directory
        self._port: int = port
        self._team: str = team
        self._input_loadout: int = input_loadout
        self._hfo: HFOEnvironment = HFOEnvironment()
        self._input_data: dict = {}

        self._num_opponents: int = -1
        self._num_teammates: int = -1
        self._team_unums: "list[int]" = []

        self._observation: np.ndarray = np.array(0)
        self._next_observation: np.ndarray = np.array(0)

        self._status: int = -1

        self._readInput()
        if setup_hfo:
            self._setupHFO()


    @property
    def status(self) -> int:
        return self._status


    def _readInput(self):
        self._input_data = readInputData(getPath(self._directory, "input"),
                                         self._inputPurpose(), self._input_loadout)
        print(f"[INFO] 'AgentForHFO.py' loaded loadout {self._input_loadout}, with the following parameters:")
        print(self._input_data)


    def _inputPurpose(self) -> str:
        return "agent"


    def _setupHFO(self):
        # Connect to the server with the specified
        # feature set. See feature sets in hfo.py/hfo.hpp.
        self._hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                                  '../HFO/bin/teams/base/config/formations-dt',
                                  self._port, 'localhost', self._team, False)

        print("[INFO] Main agent uniform number:", self._hfo.getUnum())

        self._num_opponents = self._hfo.getNumOpponents()
        self._num_teammates = self._hfo.getNumTeammates()


    def playEpisode(self) -> bool:
        self._status = IN_GAME
        self._observation = self._hfo.getState()
        self._next_observation = None

        self._atEpisodeStart()

        while self._status == IN_GAME:
            self._atTimestepStart()

            self._act(self._selectAction())

            self._status = self._hfo.step()
            self._next_observation = self._hfo.getState()

            self._atTimestepEnd()

            self._updateObservation()

        self._atEpisodeEnd()

        # Quit if the server goes down
        if self._status == SERVER_DOWN:
            self._hfo.act(QUIT)
            return False

        return True


    def _atEpisodeStart(self) -> None:
        pass


    def _atTimestepStart(self) -> None:
        pass


    def _atTimestepEnd(self) -> None:
        pass


    def _updateObservation(self) -> None:
        self._observation = self._next_observation


    def _atEpisodeEnd(self) -> None:
        pass


    @abstractmethod
    def _selectAction(self) -> int:
        pass


    def _act(self, hfo_action):
        if hfo_action == PASS:
            self._pass(self._observation)
        else:
            self._hfo.act(hfo_action)


    def _pass(self, observation):
        if not self._team_unums:
            self._team_unums = getTeamUniformNumbers(observation, self._num_teammates)

        if self._team_unums:
            self._hfo.act(PASS, self._team_unums[0])
        else:
            self._hfo.act(NOOP)
