# Adapted from https://repolinux.wordpress.com/2012/10/09/non-blocking-read-from-stdin-in-python/

from abc import abstractmethod
from threading import Thread
import time
import sys
import select

from src.hfo_agents.AgentForHFO import AgentForHFO
from src.hfo_agents.agentForHFOFactory import getAgentForHFOFactory


class WaitForQuitThread(Thread):
    def __init__(self):
        super().__init__()
        self._running = True
        self._read_list = [sys.stdin]


    def run(self):
        while self._read_list and self._running:
            ready = select.select(self._read_list, [], [], 0.1)[0]
            if ready:
                for file in ready:
                    line = file.readline()
                    if not line:  # EOF, remove file from input list
                        self._read_list.remove(file)
                    elif line.rstrip():  # optional: skipping empty lines
                        if line.lower().startswith("q"):
                            self._running = False
                            break

        print("[INFO] Thread quitting...")

    def stop(self):
        self._running = False


class AgentThread(Thread):
    def __init__(self, directory: str, port: int, input_loadout: int, agent_type: str, team: str,
                 wait_for_quit_thread: WaitForQuitThread):
        super().__init__()
        self._directory = directory
        self._port = port
        self._input_loadout = input_loadout
        self._agent_type = agent_type
        self._team = team
        self._wait_for_quit_thread = wait_for_quit_thread

    def run(self):
        time.sleep(1)
        agent: AgentForHFO = \
            getAgentForHFOFactory(self._agent_type)(self._directory, self._port, self.getTeam(), self._input_loadout)

        while self._wait_for_quit_thread.is_alive() and agent.playEpisode():
            pass

    @abstractmethod
    def getTeam(self) -> str:
        pass


class TeammateThread(AgentThread):
    def getTeam(self) -> str:
        return self._team + "_left"


class OpponentThread(AgentThread):
    def getTeam(self) -> str:
        return self._team + "_right"
