import time
import argparse
from argparse import Namespace
import sys
from typing import Type, cast, List, Optional

from src.lib.evaluation.processes import killProcesses, startProcesses
from src.lib.evaluation.storage import createOutputFiles
from src.lib.evaluation.execution import playEpisodes
from src.lib.evaluation.episodes import getEpisodeAndTrainEpisode
from src.hfo_agents.learning.LearningAgentForHFO import LearningAgentForHFO
from src.hfo_agents import is_custom, get_team
from src.hfo_agents.agentForHFOFactory import getAgentForHFOFactory
from src.lib.input import readInputData
from src.lib.io import logOutput, flushOutput, getLoadoutLabels
from src.lib.paths import getPath, getAgentStatePath, DEFAULT_DIRECTORY, DEFAULT_PORT
from src.lib.threads import WaitForQuitThread


def main() -> None:
    wait_for_quit_thread = WaitForQuitThread()
    wait_for_quit_thread.start()

    args = parseArguments()
    directory = args.directory or DEFAULT_DIRECTORY
    port = args.port or DEFAULT_PORT
    if not args.no_output:
        logOutput(getPath(directory, "output"), "w")

    print("[INFO] Starting 'seq_evaluate.py'... ('Q' + 'Return' to quit)")

    input_path = getPath(directory, "input")
    loadout_labels = getLoadoutLabels(input_path)
    num_loadouts = len(loadout_labels)

    loadouts = []
    for i in range(num_loadouts):
        read_purpose = "seq_evaluate_first_loadout" if i == 0 else "seq_evaluate"
        loadouts.append(readInputData(input_path, read_purpose, loadout_labels[i]))

    agent_type = loadouts[0]["agent_type"]
    if not is_custom(agent_type):
        exit("[ERROR] Agent must be custom.")

    agent_factory = getAgentForHFOFactory(agent_type)
    if not agent_factory.is_learning_agent():
        exit("[ERROR] Agent must be a learning agent.")

    learning_agent_factory = cast(Type[LearningAgentForHFO], agent_factory)

    sequentialEvaluateAgent(learning_agent_factory, directory, port, args, loadouts, loadout_labels,
                            wait_for_quit_thread)


def parseArguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualizer", action="store_true", help="Launch HFO visualizer.")
    parser.add_argument("-g", "--gnome-terminal", action="store_true", help="Launch agent in an external terminal.")
    parser.add_argument("-n", "--no-output", action="store_true")

    parser.add_argument("-s", "--save-period", type=int, help=f"Save data to {getAgentStatePath('')} and "
                                                              f"{getPath('', 'save')} every SAVE_PERIOD episodes.")

    parser.add_argument("-D", "--directory", type=str)
    parser.add_argument("-p", "--port", type=int)

    args = parser.parse_args()

    return args


def sequentialEvaluateAgent(agent_factory: Type[LearningAgentForHFO], directory: str, port: int, args: Namespace,
                            loadouts: List[dict], loadout_labels: List[int], wait_for_quit_thread: WaitForQuitThread):
    agent: Optional[LearningAgentForHFO] = None

    for i in range(len(loadouts)):
        input_data = loadouts[i]
        input_loadout = loadout_labels[i]
        print(f"[INFO] 'seq_evaluate.py' loaded loadout {input_loadout}, with the following parameters:")
        print(input_data)

        hfo_process = startProcesses(directory, port, args, input_loadout, input_data, wait_for_quit_thread)

        if i == 0:
            agent = agent_factory(directory, port, get_team(input_data["teammates_type"]),
                                  input_loadout, multiprocessing=True)
            createOutputFiles(directory, agent)
        else:
            teammates_type = input_data["teammates_type"]
            agent.setTeam(get_team(teammates_type))
            agent.setupHFO()
            agent.setLoadout(input_loadout)
            agent.readInput()
            agent.storeInputData(False)
            agent.setStaticParameters()
            if "reset_parameters" in input_data and input_data["reset_parameters"]:
                agent.resetParameters()

        num_episodes = {
            "test": input_data["num_test_episodes"],
            "train": input_data["num_train_episodes"],
            "total": input_data["num_test_episodes"] + input_data["num_train_episodes"],
            "save": args.save_period or 1,
            "max": input_data["max_train_episode"] or sys.maxsize
        }

        episode = getEpisodeAndTrainEpisode(directory, True, 0, num_episodes)[0]
        playEpisodes(agent, directory, episode, num_episodes, wait_for_quit_thread)

        print(f"[INFO] 'seq_evaluate.py' finished loadout {input_loadout}. Terminating associated processes...")

        agent.deleteHFO()
        killProcesses(hfo_process, args.gnome_terminal)
        time.sleep(2)

    print("[INFO] Terminating 'seq_evaluate.py'...")

    if wait_for_quit_thread.is_alive():
        wait_for_quit_thread.stop()

    if not args.no_output:
        flushOutput(getPath(directory, "output"))


if __name__ == "__main__":
    main()
