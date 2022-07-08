import argparse
import os

import torch

from src.lib.input import readInputData
from src.lib.paths import DEFAULT_DIRECTORY, getPath
from src.hfo_agents.agentForHFOFactory import getAgentForHFOFactory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--directory", type=str)
    parser.add_argument("-i", "--input-loadout", type=int)
    args = parser.parse_args()

    directory = args.directory or DEFAULT_DIRECTORY
    input_loadout = args.input_loadout or 0

    agent_type = readInputData(getPath(directory, "input"), "agent_type", input_loadout)["agent_type"]

    base_path = getPath(directory, "agent-state")
    for state in os.listdir(base_path):
        folder = base_path + "/" + state
        for file in ("model.pt", "optimizer.pt"):
            path = folder + "/" + file
            network = torch.load(path, map_location=torch.device("cpu"))
            torch.save(network, path)
            # Try to see if network is now in cpu
            torch.load(path)
            print(path, "OK")


if __name__ == "__main__":
    main()