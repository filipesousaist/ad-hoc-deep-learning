from src.hfo_agents.fixed.NoOpAgentForHFO import NoOpAgentForHFO
from src.hfo_agents.fixed.SimpleAgentForHFO import SimpleAgentForHFO
from src.hfo_agents.fixed.HeliosAgentForHFO import HeliosAgentForHFO
from src.hfo_agents.learning.DQNAgentForHFO import DQNAgentForHFO
from src.hfo_agents.learning.DRQNAgentForHFO import DRQNAgentForHFO
from src.hfo_agents.imitating.SimpleDQNAgentForHFO import SimpleDQNAgentForHFO
from src.hfo_agents.imitating.HeliosDQNAgentForHFO import HeliosDQNAgentForHFO


def getAgentForHFOFactory(agent_type: str) -> type:
    return {
        "noop": NoOpAgentForHFO,
        "simple": SimpleAgentForHFO,
        "helios": HeliosAgentForHFO,
        "dqn": DQNAgentForHFO,
        "drqn": DRQNAgentForHFO,
        "simple_dqn": SimpleDQNAgentForHFO,
        "helios_dqn": HeliosDQNAgentForHFO
    }[agent_type]
