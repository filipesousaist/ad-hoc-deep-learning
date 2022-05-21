import argparse
import os, sys

sys.path.append("./lib")
from constants import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", type=str)
    args = parser.parse_args()

    output_path = args.output_path or DEFAULT_OUTPUT_PATH
    if not os.path.exists(output_path):
        sys.exit("[ERROR]: Path \"" + output_path + "\" not found.")

    agent_state_episodes = listAgentStateEpisodes(
        output_path + "/" + AGENT_STATE_FILE_NAME,
        output_path + "/" + TEST_OUTPUT_FILE_NAME
    )

    cleanupAgentStateEpisodes(agent_state_episodes)
    

def listAgentStateEpisodes(agent_state_base_path, test_output_path):
    episodes = [int(path.lstrip("after").rstrip("episodes")) \
        for path in os.listdir(agent_state_base_path)]
    
    score_rate_per_episode = {}

    with open(test_output_path, "r") as file:
        lines = [line.rstrip("\n%").split(" ") for \
            line in file.readlines() if len(line) > 0 and line[0] == "%"]
           
        for pair in [(int(line[3]), float(line[-1])) for line in lines]:
            score_rate_per_episode[pair[0]] = pair[1]

    return sorted([(episode, score_rate_per_episode[episode]) for episode in episodes], 
        key = lambda pair: pair[0])


def cleanupAgentStateEpisodes(episodes):
    # Keep last 5 episodes, all multiples of 5000 and the 5 best of the remaining ones
    
