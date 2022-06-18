import argparse
import itertools

from src.lib.io import logOutput
from src.lib.paths import DEFAULT_DIRECTORY, DEFAULT_PORT, getPath
from src.lib.input import readInputData

from src.hfo_agents.agentForHFOFactory import getAgentForHFOFactory

def main():
    args = parseArguments()
    directory = args.directory or DEFAULT_DIRECTORY
    port = args.port or DEFAULT_PORT
    if not args.no_output:
        logOutput(getPath(directory, "output"))

    input_data = readInputData(getPath(directory, "input"), "run_agent")

    agent = getAgentForHFOFactory(input_data["agent_type"])(directory, port, "base_left")
    
    agent.setLearning(True)
    for _ in itertools.count():
        agent.playEpisode()


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--directory", type=str)
    parser.add_argument("-p", "--port", type=int)
    parser.add_argument("-n", "--no-output", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    main()