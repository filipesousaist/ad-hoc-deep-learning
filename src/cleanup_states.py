import argparse
import os
import shutil
from typing import Dict, List, Union

from src.lib.io import printTable
from src.lib.paths import DEFAULT_DIRECTORY, getPath

NUM_LAST_STATES_TO_KEEP = 5
NUM_BEST_STATES_TO_KEEP = 5
GAP_BETWEEN_STATES_TO_KEEP = 5000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--directory", type=str)
    parser.add_argument("-l", "--list-only", action="store_true")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-s", "--silent", action="store_true")
    parser.add_argument("-g", "--gap-between-states", type=int)
    args = parser.parse_args()

    directory = args.directory or DEFAULT_DIRECTORY
    if not os.path.exists(directory):
        exit("[ERROR] cleanup_states.py: Path \"" + directory + "\" not found.")

    paths = {
        "agent_state": getPath(directory, "agent-state"),
        "test_output": getPath(directory, "test-output")
    }
    for path in paths.values():
        if not os.path.exists(path):
            exit("[ERROR] cleanup_states.py: Path \"" + path + "\" not found.")

    gap_between_states_to_keep = args.gap_between_states or GAP_BETWEEN_STATES_TO_KEEP

    agent_states = listAgentStates(paths, args.silent)
    agent_states_to_delete = determineAgentStatesToDelete(agent_states, gap_between_states_to_keep, args.silent)

    if not args.list_only:
        deleteAgentStates(paths, agent_states_to_delete, args.force)


def listAgentStates(paths: Dict[str, str], silent: bool) -> List[Dict[str, Union[int, float]]]:
    states = [int(path.lstrip("after").rstrip("episodes"))
              for path in os.listdir(paths["agent_state"]) if path != "latest"]
    if not silent:
        print(states)
    score_rate_per_state = {}

    with open(paths["test_output"], "r") as file:
        lines = [line.rstrip("\n%").split(" ") for
                 line in file.readlines() if len(line) > 0 and line[0] == "%"]

        for pair in [(int(line[3]), float(line[-1])) for line in lines]:
            score_rate_per_state[pair[0]] = pair[1]

    return sorted(
        [{
            "num_episodes": state,
            "score_rate": score_rate_per_state[state]
        } for state in states if state in score_rate_per_state],
        key=lambda dict: dict["num_episodes"])


def determineAgentStatesToDelete(states: List[Dict[str, Union[int, float, str]]],
                                 gap_between_states_to_keep: int, silent: bool):
    num_states = len(states)

    if num_states <= max(NUM_BEST_STATES_TO_KEEP, NUM_LAST_STATES_TO_KEEP):
        return []

    min_num_episodes = states[-NUM_LAST_STATES_TO_KEEP]["num_episodes"]
    min_score_rate = sorted(states, key=lambda dict: dict["score_rate"])[-NUM_BEST_STATES_TO_KEEP]["score_rate"]

    states_to_delete = []
    states_to_keep = []

    for state in states:
        num_episodes = state["num_episodes"]
        score_rate = state["score_rate"]

        reasons_to_keep = []
        if num_episodes >= min_num_episodes:
            reasons_to_keep.append("last {} states".format(NUM_LAST_STATES_TO_KEEP))
        if num_episodes % gap_between_states_to_keep == 0:
            reasons_to_keep.append("num_episodes multiple of {}".format(gap_between_states_to_keep))
        if score_rate >= min_score_rate and score_rate > 0:
            reasons_to_keep.append("best {} states".format(NUM_BEST_STATES_TO_KEEP))

        if reasons_to_keep:
            state["reason"] = " & ".join(reasons_to_keep)
            states_to_keep.append(state)
        else:
            states_to_delete.append(state)

    if not silent:
        print("*** Agent states to keep ***")
        printTable(states_to_keep, ["num_episodes", "score_rate", "reason"])

    return states_to_delete


def deleteAgentStates(paths: Dict[str, str], states: List[Dict[str, Union[int, float, str]]],
                      force: bool):
    # Keep last 5 states, all multiples of 5000 and the 5 best overall. 
    # In case of tie, keep all the tied states, unless they have a score rate of 0.
    # In that case, keep none of them.

    if not force and not input("\n*** Confirm cleanup? (y/n) ***\n>>> ").lower().startswith("y"):
        print("State cleanup cancelled.")
        return

    print("Cleaning up {} states...".format(len(states)))

    for state in states:
        full_state_path = paths["agent_state"] + "/after{}episodes".format(state["num_episodes"])
        print("Cleaning up state \"{}\"...".format(full_state_path), end="\r")
        shutil.rmtree(full_state_path)

    print("\nCleanup done!")


if __name__ == '__main__':
    main()
