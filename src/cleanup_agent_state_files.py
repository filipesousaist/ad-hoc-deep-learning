import argparse
import os, sys, shutil

from src.lib.paths import DEFAULT_DIRECTORY, getPath
from src.lib.io import printTable


NUM_LAST_STATES_TO_KEEP = 5
NUM_BEST_STATES_TO_KEEP = 5
GAP_BETWEEN_STATES_TO_KEEP = 5000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--directory", type=str)
    parser.add_argument("-l", "--list-only", action="store_true")
    args = parser.parse_args()

    directory = args.directory or DEFAULT_DIRECTORY
    if not os.path.exists(directory):
        sys.exit("[ERROR]: Path \"" + directory + "\" not found.")

    paths = {
        "agent_state": getPath(directory, "agent-state"),
        "test_output": getPath(directory, "test-output")
    }

    agent_states = listAgentStates(paths)
    agent_states_to_delete = determineAgentStatesToDelete(agent_states)

    if not args.list_only:
        deleteAgentStates(paths, agent_states_to_delete)
    

def listAgentStates(paths):
    states = [int(path.lstrip("after").rstrip("episodes")) \
        for path in os.listdir(paths["agent_state"])]
    
    score_rate_per_state = {}

    with open(paths["test_output"], "r") as file:
        lines = [line.rstrip("\n%").split(" ") for \
            line in file.readlines() if len(line) > 0 and line[0] == "%"]
           
        for pair in [(int(line[3]), float(line[-1])) for line in lines]:
            score_rate_per_state[pair[0]] = pair[1]

    return sorted(
        [{
            "num_episodes": state, 
            "score_rate": score_rate_per_state[state]
        } for state in states], 
        key = lambda dict: dict["num_episodes"])


def determineAgentStatesToDelete(states):
    num_states = len(states)

    if num_states <= max(NUM_BEST_STATES_TO_KEEP, NUM_LAST_STATES_TO_KEEP):
        return

    min_num_episodes = states[-NUM_LAST_STATES_TO_KEEP]["num_episodes"]
    min_score_rate = sorted(states, key = lambda dict: dict["score_rate"])[-NUM_BEST_STATES_TO_KEEP]["score_rate"]

    states_to_delete = []
    states_to_keep = []

    for state in states:
        num_episodes = state["num_episodes"]
        score_rate = state["score_rate"]

        reasons_to_keep = []
        if num_episodes >= min_num_episodes:
            reasons_to_keep.append("last {} states".format(NUM_LAST_STATES_TO_KEEP))
        if num_episodes % GAP_BETWEEN_STATES_TO_KEEP == 0:
            reasons_to_keep.append("num_episodes muliple of {}".format(GAP_BETWEEN_STATES_TO_KEEP))
        if score_rate >= min_score_rate and score_rate > 0:
            reasons_to_keep.append("best {} states".format(NUM_BEST_STATES_TO_KEEP))

        if reasons_to_keep:
            state["reason"] = " & ".join(reasons_to_keep)
            states_to_keep.append(state)
        else:
            states_to_delete.append(state)
    
    print("*** Agent states to keep ***")
    printTable(states_to_keep, ["num_episodes", "score_rate", "reason"])

    return states_to_delete


def deleteAgentStates(paths, states):
    # Keep last 5 states, all multiples of 5000 and the 5 best overall. 
    # In case of tie, keep all of the tied states, unless they have a score rate of 0.
    # In that case, keep none of them.
    
    if not input("\n*** Confirm cleanup? (y/n) ***\n>>> ").lower().startswith("y"):
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
