import argparse
import os, sys

sys.path.append("./lib")
from constants import *

NUM_LAST_STATES_TO_KEEP = 5
NUM_BEST_STATES_TO_KEEP = 5
GAP_BETWEEN_STATES_TO_KEEP = 5000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", type=str)
    args = parser.parse_args()

    output_path = args.output_path or DEFAULT_OUTPUT_PATH
    if not os.path.exists(output_path):
        sys.exit("[ERROR]: Path \"" + output_path + "\" not found.")

    agent_states = listAgentStates(
        output_path + "/" + AGENT_STATE_FILE_NAME,
        output_path + "/" + TEST_OUTPUT_FILE_NAME
    )

    cleanupAgentStates(agent_states)
    

def listAgentStates(agent_state_base_path, test_output_path):
    states = [int(path.lstrip("after").rstrip("episodes")) \
        for path in os.listdir(agent_state_base_path)]
    
    score_rate_per_state = {}

    with open(test_output_path, "r") as file:
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


def cleanupAgentStates(states):
    # Keep last 5 states, all multiples of 5000 and the 5 best overall. 
    # In case of tie, keep all of the tied states, unless they have a score rate of 0.
    # In that case, keep none of them.

    num_states = len(states)

    if num_states <= max(NUM_BEST_STATES_TO_KEEP, NUM_LAST_STATES_TO_KEEP):
        return

    min_num_episodes = states[-NUM_LAST_STATES_TO_KEEP]["num_episodes"]
    min_score_rate = sorted(states, key = lambda dict: dict["score_rate"])[-NUM_BEST_STATES_TO_KEEP]["score_rate"]

    states_to_keep = []

    for state in states:
        num_episodes = state["num_episodes"]
        score_rate = state["score_rate"]
        if num_episodes >= min_num_episodes or \
           num_episodes % GAP_BETWEEN_STATES_TO_KEEP == 0 or \
           (score_rate >= min_score_rate and score_rate > 0):

            states_to_keep.append(state) 
            print(state)

    
if __name__ == '__main__':
    main()