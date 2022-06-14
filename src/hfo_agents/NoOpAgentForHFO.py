from hfo import NOOP

from src.hfo_agents.AgentForHFO import AgentForHFO

class NoOpAgentForHFO(AgentForHFO):   
    def _selectAction(self) -> int:
        return NOOP