from src.hfo_agents.AgentForHFO import AgentForHFO

from src.lib.helios_policy import HeliosPolicy


class HeliosAgentForHFO(AgentForHFO):
    def __init__(self, directory: str, port: int = -1, team: str = "", setup_hfo: bool = True):
        super().__init__(directory, port, team, setup_hfo)
        self._policy: HeliosPolicy = HeliosPolicy()


    def _atTimestepStart(self) -> None:
        self._policy.reset()


    def _selectAction(self):
        return self._policy.get_action(self._observation, self._num_teammates, self._num_opponents)
