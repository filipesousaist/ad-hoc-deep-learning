import os
import random
import time
from argparse import Namespace, ArgumentParser
from subprocess import Popen
from typing import cast, Type, Optional

import numpy as np

from src.hfo_agents import get_team
from src.hfo_agents.agentForHFOFactory import getAgentForHFOFactory
from src.hfo_agents.plastic.PLASTICAgentForHFO import PLASTICAgentForHFO
from src.lib.evaluation.execution import playTestEpisodes
from src.lib.evaluation.processes import startProcesses, killProcesses
from src.lib.input import readInputData
from src.lib.io import readJSON, writeJSON
from src.lib.paths import DEFAULT_DIRECTORY, DEFAULT_PORT, getPath
from src.lib.threads import WaitForQuitThread


def main() -> None:
    wait_for_quit_thread = WaitForQuitThread()
    wait_for_quit_thread.start()

    args = parseArguments()
    directory = args.directory or DEFAULT_DIRECTORY
    port = args.port or DEFAULT_PORT

    print("[INFO] Starting 'evaluate_plastic.py'... ('Q' + 'Return' to quit)")

    input_loadout = args.input_loadout or 0
    input_data = readInputData(getPath(directory, "input"), "evaluate_plastic", input_loadout)
    print(f"[INFO] 'evaluate_plastic.py' loaded loadout {input_loadout}, with the following parameters:")
    print(input_data)

    agent_type = input_data["agent_type"]
    agent_factory = getAgentForHFOFactory(agent_type)
    if not issubclass(agent_factory, PLASTICAgentForHFO):
        exit("[ERROR] Agent must be a PLASTIC agent.")

    plastic_agent_factory = cast(Type[PLASTICAgentForHFO], agent_factory)
    evaluatePlastic(plastic_agent_factory, directory, port, args, input_data, input_loadout, wait_for_quit_thread,
                    input_data["num_trials"], input_data["num_episodes_per_trial"])


def evaluatePlastic(agent_factory: Type[PLASTICAgentForHFO], directory: str, port: int, args: Namespace,
                    input_data: dict, input_loadout: int, wait_for_quit_thread: WaitForQuitThread, num_trials: int,
                    num_episodes_per_trial: int):

    agent: Optional[PLASTICAgentForHFO] = None

    first_trial = getNextTrial(directory, input_loadout)

    for trial in range(first_trial, num_trials + 1):
        team = random.choice(input_data["possible_teams"])
        teammates_type = "bin_" + team
        hfo_process = startProcesses(directory, port, args, input_loadout, input_data, wait_for_quit_thread,
                                     teammates_type)

        if trial == first_trial:
            agent = agent_factory(directory, args.teams_directory, input_data["eta"], port, get_team(teammates_type),
                                  input_loadout, multiprocessing=True)
        else:
            agent.setTeam(get_team(teammates_type))
            agent.setupHFO()

        agent.reset()

        episode, server_up = playTestEpisodes(agent, num_episodes_per_trial + 1, wait_for_quit_thread)
        while not server_up and wait_for_quit_thread.is_alive():
            shutdownAndWait(agent, hfo_process, args.gnome_terminal, 3)
            hfo_process = startProcesses(directory, port, args, input_loadout, input_data, wait_for_quit_thread,
                                         teammates_type)
            agent.setupHFO()
            episode, server_up = playTestEpisodes(agent, num_episodes_per_trial + 1, wait_for_quit_thread, episode)


        if not wait_for_quit_thread.is_alive():
            shutdownAndWait(agent, hfo_process, args.gnome_terminal, 0)
            break

        agent.results["correct_team"] = team
        agent.results["guessed_team"] = input_data["known_teams"][np.argmax(agent.results["behavior_distribution"][-1])]
        print(f"[INFO] 'evaluate_plastic.py' finished trial {trial}. Terminating associated processes...")
        saveResults(directory, input_loadout, trial, agent.results)

        shutdownAndWait(agent, hfo_process, args.gnome_terminal, 3)

    print("[INFO] Terminating 'evaluate_plastic.py'...")

    if wait_for_quit_thread.is_alive():
        wait_for_quit_thread.stop()


def shutdownAndWait(agent: PLASTICAgentForHFO, hfo_process: Popen, gnome_terminal: bool, duration: int):
    agent.deleteHFO()
    killProcesses(hfo_process, gnome_terminal)
    time.sleep(duration)


def getNextTrial(directory: str, input_loadout: int) -> int:
    results_directory = os.path.join(directory, str(input_loadout))
    if not os.path.exists(results_directory):
        os.mkdir(results_directory)

    results_path = getPath(results_directory, "plastic-results")
    if not os.path.exists(results_path):
        writeJSON(results_path, {})

    results = readJSON(results_path)
    trials = [int(trial_str) for trial_str in results.keys()]
    return max(trials) + 1 if len(trials) > 0 else 1


def saveResults(directory: str, input_loadout: int, trial: int, new_results: dict) -> None:
    results_path = getPath(os.path.join(directory, str(input_loadout)), "plastic-results")
    results = readJSON(results_path)
    results[str(trial)] = new_results
    writeJSON(results_path, results)


def parseArguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-v", "--visualizer", action="store_true", help="Launch HFO visualizer.")
    parser.add_argument("-g", "--gnome-terminal", action="store_true", help="Launch agent in an external terminal.")

    parser.add_argument("-D", "--directory", type=str)
    parser.add_argument("-T", "--teams-directory", type=str)
    parser.add_argument("-p", "--port", type=int)
    parser.add_argument("-i", "--input-loadout", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    main()
