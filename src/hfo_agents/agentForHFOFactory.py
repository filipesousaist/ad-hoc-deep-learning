from src.hfo_agents.DQNAgentForHFO import DQNAgentForHFO
from src.hfo_agents.DRQNAgentForHFO import DRQNAgentForHFO
from src.hfo_agents.SimpleAgentForHFO import SimpleAgentForHFO
from src.hfo_agents.NoOpAgentForHFO import NoOpAgentForHFO

def getAgentForHFOFactory(agent_type: str) -> type: 
    return {
        "noop": NoOpAgentForHFO,
        "simple": SimpleAgentForHFO,
        "dqn": DQNAgentForHFO,
        "drqn": DRQNAgentForHFO
    }[agent_type]