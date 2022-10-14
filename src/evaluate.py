import argparse
from argparse import Namespace
import sys

from typing import Type, cast


from src.lib.paths import DEFAULT_DIRECTORY, DEFAULT_PORT, getPath, getAgentStatePath
from src.lib.io import logOutput, flushOutput
from src.lib.threads import WaitForQuitThread
from src.lib.input import readInputData
from src.lib.evaluation.episodes import getEpisodeAndTrainEpisode, getBestTrainEpisode
from src.lib.evaluation.storage import createOutputFiles, loadAgent
from src.lib.evaluation.processes import killProcesses, startProcesses
from src.lib.evaluation.execution import playTestEpisodes, playEpisodes

from src.hfo_agents import is_custom, get_team_name
from src.hfo_agents.agentForHFOFactory import getAgentForHFOFactory
from src.hfo_agents.AgentForHFO import AgentForHFO
from src.hfo_agents.learning.LearningAgentForHFO import LearningAgentForHFO


def main() -> None:
    wait_for_quit_thread = WaitForQuitThread()
    wait_for_quit_thread.start()

    args = parseArguments()
    directory = args.directory or DEFAULT_DIRECTORY
    port = args.port or DEFAULT_PORT
    if not args.no_output:
        open_mode = "a" if (args.load or args.load_best or args.test_from_episode) else "w"
        logOutput(getPath(directory, "output"), open_mode)

    print("[INFO] Starting 'evaluate.py'... ('Q' + 'Return' to quit)")

    input_loadout = args.input_loadout or 0
    input_data = readInputData(getPath(directory, "input"), "evaluate", input_loadout)
    print(f"[INFO] 'evaluate.py' loaded loadout {input_loadout}, with the following parameters:")
    print(input_data)

    hfo_process = startProcesses(directory, port, args, input_loadout, input_data, wait_for_quit_thread)

    agent_type = input_data["agent_type"]
    if is_custom(agent_type):
        teammates_type = input_data["teammates_type"]
        team_name = ("base" if is_custom(teammates_type) else get_team_name(teammates_type)) + "_left"
        agent_factory: Type[AgentForHFO] = getAgentForHFOFactory(agent_type)
        if agent_factory.is_learning_agent():
            learning_agent_factory = cast(Type[LearningAgentForHFO], agent_factory)
            learning_agent = learning_agent_factory(
                directory, port, team_name, input_loadout,
                load_parameters=(args.load or args.load_best) and not args.reset_parameters
            )
            evaluateAgent(learning_agent, directory, args, input_data, wait_for_quit_thread)
            learning_agent.deleteHFO()
        else:
            agent = agent_factory(directory, port, team_name, input_loadout)
            playTestEpisodes(agent, args.max_episode or sys.maxsize, wait_for_quit_thread)
            agent.deleteHFO()
    else:
        while wait_for_quit_thread.is_alive():
            pass

    print("[INFO] Terminating 'evaluate.py' and associated processes...")

    if wait_for_quit_thread.is_alive():
        wait_for_quit_thread.stop()

    if not args.no_output:
        flushOutput(getPath(directory, "output"))

    killProcesses(hfo_process, args.gnome_terminal)


def parseArguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualizer", action="store_true", help="Launch HFO visualizer.")
    parser.add_argument("-g", "--gnome-terminal", action="store_true", help="Launch agent in an external terminal.")
    parser.add_argument("-n", "--no-output", action="store_true")

    parser.add_argument("-l", "--load", action="store_true",
                        help=f"Load data stored in {getAgentStatePath('')} and {getPath('', 'save')}.")
    parser.add_argument("-L", "--load-best", action="store_true",
                        help=f"Load agent state with highest score rate and data stored in {getPath('', 'save')}.")
    parser.add_argument("-s", "--save-period", type=int, help=f"Save data to {getAgentStatePath('')} and "
                                                              f"{getPath('', 'save')} every SAVE_PERIOD episodes.")

    parser.add_argument("-t", "--test-from-episode", type=int)
    parser.add_argument("-m", "--max-episode", type=int, help="Stop running at train episode MAX_EPISODE in "
                                                              "train/test mode, or at test episode MAX_EPISODE in "
                                                              "test mode.")
    parser.add_argument("-D", "--directory", type=str)
    parser.add_argument("-p", "--port", type=int)
    parser.add_argument("-i", "--input-loadout", type=int)
    parser.add_argument("-r", "--reset-parameters", action="store_true",
                        help=f"Use parameters as defined in loadout instead of loading saved ones.")

    args = parser.parse_args()

    if args.test_from_episode:
        if args.load:
            exit("'load' and 'test-from-episode' cannot be both set.")
        elif args.load_best:
            exit("'load-best' and 'test-from-episode' cannot be both set.")
        elif args.save_period:
            exit("'save-period' and 'test-from-episode' cannot be both set.")

    if args.load and args.load_best:
        exit("'load' and 'load-best' cannot be both set.")

    return args


def evaluateAgent(agent: LearningAgentForHFO, directory: str, args: Namespace, input_data: dict,
                  wait_for_quit_thread: WaitForQuitThread) -> None:
    num_episodes = {
        "test": input_data["num_test_episodes"],
        "train": input_data["num_train_episodes"],
        "total": input_data["num_test_episodes"] + input_data["num_train_episodes"],
        "save": args.save_period or 1,
        "max": args.max_episode or sys.maxsize
    }
    episode, train_episode = getEpisodeAndTrainEpisode(
        directory, args.load or args.load_best, args.test_from_episode, num_episodes)

    if args.load or args.test_from_episode == -1:
        loadAgent(agent, directory, -1, num_episodes)
    elif args.load_best:
        loadAgent(agent, directory, getBestTrainEpisode(directory), num_episodes)
    elif args.test_from_episode:
        loadAgent(agent, directory, train_episode, num_episodes)
    else:
        createOutputFiles(directory, agent)

    if args.test_from_episode:
        agent.setLearning(False)
        playTestEpisodes(agent, num_episodes["max"], wait_for_quit_thread)
    else:
        playEpisodes(agent, directory, episode, num_episodes, wait_for_quit_thread)


if __name__ == '__main__':
    main()
