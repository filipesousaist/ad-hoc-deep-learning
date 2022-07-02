from src.hfo_agents.learning.DQNAgentForHFO import DQNAgentForHFO
from src.hfo_agents.learning.DRQNAgentForHFO import DRQNAgentForHFO
from src.hfo_agents.imitating.SimpleDQNAgentForHFO import SimpleDQNAgentForHFO
from src.hfo_agents.imitating.HeliosDQNAgentForHFO import HeliosDQNAgentForHFO
from src.hfo_agents.fixed.SimpleAgentForHFO import SimpleAgentForHFO
from src.hfo_agents.fixed.NoOpAgentForHFO import NoOpAgentForHFO


def getAgentForHFOFactory(agent_type: str) -> type:
    return {
        "noop": NoOpAgentForHFO,
        "simple": SimpleAgentForHFO,
        "dqn": DQNAgentForHFO,
        "simple_dqn": SimpleDQNAgentForHFO,
        "helios_dqn": HeliosDQNAgentForHFO,
        "drqn": DRQNAgentForHFO
    }[agent_type]
