from typing import Type

from src.hfo_agents.AgentForHFO import AgentForHFO
from src.hfo_agents.fixed.NoOpAgentForHFO import NoOpAgentForHFO
from src.hfo_agents.fixed.SimpleAgentForHFO import SimpleAgentForHFO
from src.hfo_agents.fixed.HeliosAgentForHFO import HeliosAgentForHFO
from src.hfo_agents.learning.MLPDQNAgentForHFO import MLPDQNAgentForHFO
from src.hfo_agents.learning.DRQNAgentForHFO import DRQNAgentForHFO
from src.hfo_agents.imitating.SimpleDQNAgentForHFO import SimpleDQNAgentForHFO
from src.hfo_agents.imitating.HeliosDQNAgentForHFO import HeliosDQNAgentForHFO
from src.hfo_agents.plastic.MLPDQNPLASTICAgentForHFO import MLPDQNPLASTICAgentForHFO
from src.hfo_agents.plastic.DRQNPLASTICAgentForHFO import DRQNPLASTICAgentForHFO


def getAgentForHFOFactory(agent_type: str) -> Type[AgentForHFO]:
    return {
        "noop": NoOpAgentForHFO,
        "simple": SimpleAgentForHFO,
        "helios": HeliosAgentForHFO,
        "dqn": MLPDQNAgentForHFO,
        "drqn": DRQNAgentForHFO,
        "simple_dqn": SimpleDQNAgentForHFO,
        "helios_dqn": HeliosDQNAgentForHFO,
        "dqn_plastic": MLPDQNPLASTICAgentForHFO,
        "drqn_plastic": DRQNPLASTICAgentForHFO
    }[agent_type]
