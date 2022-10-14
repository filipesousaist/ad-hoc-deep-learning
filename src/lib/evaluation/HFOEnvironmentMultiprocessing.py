from multiprocessing import Process, JoinableQueue
from typing import Any

import numpy as np
from hfo import HFOEnvironment, QUIT


class HFOEnvironmentMultiprocessing(HFOEnvironment):
    def __init__(self):
        super().__init__()
        self._agent_queue = JoinableQueue(1)
        self._hfo_queue = JoinableQueue(1)
        _HFOInterfaceProcess(self._agent_queue, self._hfo_queue).start()

    def _get(self, name: str) -> Any:
        self._agent_queue.put("GET_" + name)
        return self._hfo_queue.get()

    def _execute(self, name: str, *args) -> None:
        self._agent_queue.put(name)
        if len(args) > 0:
            self._agent_queue.put(args)

    def connectToServer(self, *args) -> None:
        self._execute("CONNECT_TO_SERVER", *args)

    def getNumTeammates(self) -> int:
        return self._get("NUM_TEAMMATES")

    def getNumOpponents(self) -> int:
        return self._get("NUM_OPPONENTS")

    def getUnum(self) -> int:
        return self._get("UNUM")

    def act(self, *args) -> None:
        self._execute("ACT", *args)

    def getState(self, state_data=None) -> int:
        return self._get("STATE")

    def step(self) -> np.ndarray:
        return self._get("STEP")

    def __del__(self):
        self._execute("QUIT")
        super().__del__()


class _HFOInterfaceProcess(Process):
    def __init__(self, agent_queue: JoinableQueue, hfo_queue: JoinableQueue):
        super().__init__()
        self._agent_queue = agent_queue
        self._hfo_queue = hfo_queue
        self._hfo = HFOEnvironment()

    def run(self) -> None:
        while self._handleRequest(self._agent_queue.get()):
            pass


    def _handleRequest(self, request: str) -> bool:
        if request == "CONNECT_TO_SERVER":
            self._hfo.connectToServer(*self._agent_queue.get())
        elif request == "GET_NUM_OPPONENTS":
            self._hfo_queue.put(self._hfo.getNumOpponents())
        elif request == "GET_NUM_TEAMMATES":
            self._hfo_queue.put(self._hfo.getNumTeammates())
        elif request == "GET_UNUM":
            self._hfo_queue.put(self._hfo.getUnum())
        elif request == "GET_STATE":
            self._hfo_queue.put(self._hfo.getState())
        elif request == "GET_STEP":
            self._hfo_queue.put(self._hfo.step())
        elif request == "ACT":
            args = self._agent_queue.get()
            self._hfo.act(*args)
            if args[0] == QUIT:
                return False
        elif request == "QUIT":
            return False
        return True
