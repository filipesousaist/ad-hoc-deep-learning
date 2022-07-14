from src.hfo_agents.learning.MLPDQNAgentForHFO import MLPDQNAgentForHFO

from src.lib.helios_policy import HeliosPolicy


class HeliosDQNAgentForHFO(MLPDQNAgentForHFO):
    def __init__(self, directory: str, port: int = -1, team: str = "", setup_hfo: bool = True):
        super().__init__(directory, port, team, setup_hfo)
        self._policy: HeliosPolicy = HeliosPolicy()

    def _atTimestepStart(self) -> None:
        self._policy.reset()

    def _getAction(self):
        return self._policy.get_action(self._observation, self._num_teammates, self._num_opponents)
