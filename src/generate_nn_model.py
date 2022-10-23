from argparse import ArgumentParser, Namespace
from typing import Type, cast

from src.lib.evaluation.processes import startProcesses, killProcesses
from src.lib.evaluation.storage import loadAgent
from src.hfo_agents.learning.LearningAgentForHFO import LearningAgentForHFO
from src.hfo_agents.agentForHFOFactory import getAgentForHFOFactory
from src.hfo_agents import is_custom, get_team_name
from src.lib.evaluation.episodes import getBestTrainEpisode
from src.lib.input import readInputData
from src.lib.io import getLoadoutLabels
from src.lib.paths import DEFAULT_DIRECTORY, getPath, DEFAULT_PORT
from src.lib.threads import WaitForQuitThread


def main() -> None:
    wait_for_quit_thread = WaitForQuitThread()
    wait_for_quit_thread.start()

    args = parseArguments()
    directory = args.directory or DEFAULT_DIRECTORY
    port = args.port or DEFAULT_PORT
    train_episode = getBestTrainEpisode(directory) if args.best_train_episode else \
        (args.train_episode or -1)

    input_path = getPath(directory, "input")
    last_label = getLoadoutLabels(input_path)[-1]
    input_data = readInputData(input_path, "evaluate", last_label)

    args.gnome_terminal = False
    args.visualizer = False
    hfo_process = startProcesses(directory, port, args, last_label, input_data, wait_for_quit_thread)

    agent_type = input_data["agent_type"]
    if not is_custom(agent_type):
        exit("[ERROR] Agent must be custom.")

    agent_factory = getAgentForHFOFactory(agent_type)
    if not agent_factory.is_learning_agent():
        exit("[ERROR] Agent must be a learning agent.")

    learning_agent_factory = cast(Type[LearningAgentForHFO], agent_factory)

    teammates_type = input_data["teammates_type"]
    team_name = ("base" if is_custom(teammates_type) else get_team_name(teammates_type)) + "_left"
    agent = learning_agent_factory(directory, port, team_name, last_label)

    loadAgent(agent, directory, train_episode, input_data["num_train_episodes"])

    agent.createNNModel(train_episode)

    if wait_for_quit_thread.is_alive():
        wait_for_quit_thread.stop()

    killProcesses(hfo_process, False)


def parseArguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-D", "--directory", type=str)
    parser.add_argument("-t", "--train-episode", type=int)
    parser.add_argument("-b", "--best-train-episode", action="store_true")
    parser.add_argument("-p", "--port", type=int)

    args = parser.parse_args()

    if args.train_episode and args.best_train_episode:
        exit("'train-episode' and 'best-train-episode' cannot be both set.")

    return args


if __name__ == "__main__":
    main()
