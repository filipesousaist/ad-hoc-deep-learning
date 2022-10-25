from src.lib.io import readJSON

_REQUIRED_ARGS = {
    "evaluate": [
        "agent_type", "fullstate", "num_teammates", "num_opponents", "teammates_type", "opponents_type",
        "frames_per_trial", "untouched_time", "num_test_episodes", "num_train_episodes"
    ],
    "seq_evaluate_first_loadout": [
        "agent_type", "fullstate", "num_teammates", "num_opponents", "teammates_type", "opponents_type",
        "frames_per_trial", "untouched_time", "num_test_episodes", "num_train_episodes",
        "max_train_episode"
    ],
    "seq_evaluate": [
        "agent_type", "fullstate", "num_teammates", "num_opponents", "teammates_type", "opponents_type",
        "frames_per_trial", "untouched_time", "num_test_episodes", "num_train_episodes",
        "max_train_episode"
    ],
    "agent": [],
    "learning_agent": [
        "agent_parameters", "actions", "reward_function"
    ],
    "agent_type": [
        "agent_type"
    ],
    "generate_nn_model": [
        "agent_type", "num_train_episodes"
    ],
    "plastic_agent": [
        "agent_parameters", "actions", "reward_function", "known_teams"
    ],
    "evaluate_plastic": [
        "agent_type", "fullstate", "num_teammates", "num_opponents", "known_teams", "possible_teams", "opponents_type",
        "frames_per_trial", "untouched_time", "num_episodes_per_trial", "num_trials", "eta"
    ]
}

_OPTIONAL_ARGS = {
    "evaluate": [],
    "seq_evaluate_first_loadout": [],
    "seq_evaluate": ["reset_parameters"],
    "agent": [],
    "learning_agent": [
        "ignore_auto_move_chance", "see_move_period", "feature_extractors", "filter_policy"
    ],
    "plastic_agent": [
        "feature_extractors", "filter_policy"
    ],
    "agent_type": [],
    "generate_nn_model": [],
    "evaluate_plastic": []
}

_REQUIRED_ARGS["imitating_agent"] = _REQUIRED_ARGS["learning_agent"] + \
    ["initial_chance_to_imitate", "final_chance_to_imitate", "steps_to_imitate"]


def readInputData(path: str, purpose: str, loadout: int = 0) -> dict:
    input_dict = readJSON(path)

    input_data = input_dict[str(loadout)] if loadout > 0 else input_dict

    missing_args = [arg for arg in _REQUIRED_ARGS[purpose] if arg not in input_data]

    if len(missing_args) > 0:
        print(f"[ERROR]: Missing required argument(s): {', '.join(missing_args)}")

    args_to_delete = [arg for arg in input_data
                      if (arg not in _REQUIRED_ARGS[purpose]) and (arg not in _OPTIONAL_ARGS[purpose])]
    for arg in args_to_delete:
        del input_data[arg]

    return input_data


