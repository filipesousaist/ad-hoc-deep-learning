from abc import abstractclassmethod
from threading import Thread
from typing import Type
import time

from src.hfo_agents.AgentForHFO import AgentForHFO
from src.hfo_agents.agentForHFOFactory import getAgentForHFOFactory


class WaitForQuitThread(Thread):
    def run(self):
        while not input().lower().startswith('q'):
            pass


class AgentThread(Thread):
    def __init__(self, directory: str, port: int, input_loadout: int, agent_type: str, wait_for_quit_thread: WaitForQuitThread):
        super().__init__()
        self._directory = directory
        self._port = port
        self._input_loadout = input_loadout
        self._agent_type = agent_type
        self._wait_for_quit_thread = wait_for_quit_thread

    def run(self):
        time.sleep(1)
        agent: Type[AgentForHFO] = \
            getAgentForHFOFactory(self._agent_type)(self._directory, self._port, self.getTeam(), self._input_loadout)
        
        while self._wait_for_quit_thread.is_alive() and agent.playEpisode():
            pass
    
    @abstractclassmethod
    def getTeam(self) -> str:
        pass


class TeammateThread(AgentThread):
    def getTeam(self) -> str:
        return "base_left"


class OpponentThread(AgentThread):
    def getTeam(self) -> str:
        return "base_right"
