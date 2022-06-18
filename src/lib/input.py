from src.lib.io import readJSON


_REQUIRED_ARGS = {
    "evaluate": [
        "agent_type", "fullstate", "num_teammates", "num_opponents", "teammates_type", "opponents_type",
        "frames_per_trial", "untouched_time", "num_test_episodes", "num_train_episodes"
    ],
    "agent": [],
    "learning_agent": [
        "agent_parameters", "actions", "custom_features", "reward_function"
    ],
    "run_agent": [
        "agent_type"
    ]
}


def readInputData(path: str, purpose: str, loadout: int = 0) -> dict:
    input_dict = readJSON(path)

    input_data = input_dict[str(loadout)] if loadout > 0 else input_dict
    
    missing_args = [arg for arg in _REQUIRED_ARGS[purpose] if arg not in input_data]
    
    if len(missing_args) > 0:
        print(f"[ERROR]: Missing required argument(s): {', '.join(missing_args)}")

    unnecessary_args = [arg for arg in input_data if arg not in _REQUIRED_ARGS[purpose]]
    for arg in unnecessary_args:
        del input_data[arg]

    return input_data