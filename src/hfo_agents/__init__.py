def is_custom(agent_type: str) -> bool:
    return agent_type != "npc" and not agent_type.startswith("bin_")


def get_team_name(agent_type: str) -> str:
    return agent_type[4:] if agent_type.startswith("bin_") else "base"


def get_team(teammates_type: str) -> str:
    return ("base" if is_custom(teammates_type) else get_team_name(teammates_type)) + "_left"
