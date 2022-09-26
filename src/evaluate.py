import argparse
import os
import sys

import psutil
from psutil import Popen
from signal import SIGTERM
from threading import Thread
import time
from typing import Type, cast

import hfo
from hfo import GOAL

from src.lib.paths import DEFAULT_DIRECTORY, DEFAULT_PORT, getPath, getAgentStatePath
from src.lib.io import logOutput, flushOutput, readTxt, writeTxt
from src.lib.time import getReadableTime
from src.lib.threads import WaitForQuitThread, TeammateThread, OpponentThread
from src.lib.input import readInputData

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
        open_mode = "a" if (args.load or args.test_from_episode) else "w"
        logOutput(getPath(directory, "output"), open_mode)

    print("[INFO] Starting 'evaluate.py'... ('Q' + 'Return' to quit)")

    input_loadout = args.input_loadout or 0
    input_data = readInputData(getPath(directory, "input"), "evaluate", input_loadout)
    print(f"[INFO] 'evaluate.py' loaded loadout {input_loadout}, with the following parameters:")
    print(input_data)

    hfo_process = launchHFO(input_data, port, args.gnome_terminal, args.visualizer)
    time.sleep(2)
    launchOtherAgents(directory, port, input_loadout, input_data, wait_for_quit_thread)

    agent_type = input_data["agent_type"]
    if is_custom(agent_type):
        teammates_type = input_data["teammates_type"]
        team_name = ("base" if is_custom(teammates_type) else get_team_name(teammates_type)) + "_left"
        agent_factory: Type[AgentForHFO] = getAgentForHFOFactory(agent_type)
        if agent_factory.is_learning_agent():
            learning_agent_factory = cast(Type[LearningAgentForHFO], agent_factory)
            learning_agent = learning_agent_factory(directory, port, team_name, input_loadout,
                                                    load_parameters=args.load)
            evaluateAgent(learning_agent, directory, args, input_data, wait_for_quit_thread)
        else:
            agent = agent_factory(directory, port, team_name, input_loadout)
            playTestEpisodes(agent, wait_for_quit_thread)
    else:
        while wait_for_quit_thread.is_alive():
            pass

    print("[INFO] Terminating 'evaluate.py' and associated processes...")

    if not args.no_output:
        flushOutput(getPath(directory, "output"))

    killProcesses(hfo_process, args.gnome_terminal)


def is_custom(agent_type: str) -> bool:
    return agent_type != "npc" and not agent_type.startswith("bin_")


def get_team_name(agent_type: str) -> str:
    return agent_type[4:] if agent_type.startswith("bin_") else "base"


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualizer", action="store_true", help="Launch HFO visualizer.")
    parser.add_argument("-g", "--gnome-terminal", action="store_true", help="Launch agent in an external terminal.")
    parser.add_argument("-n", "--no-output", action="store_true")

    parser.add_argument("-l", "--load", action="store_true",
                        help=f"Load data stored in {getAgentStatePath('')} and {getPath('', 'save')}.")
    parser.add_argument("-s", "--save-period", type=int, help=f"Save data to {getAgentStatePath('')} and "
                                                              f"{getPath('', 'save')} every SAVE_PERIOD episodes.")
    parser.add_argument("-t", "--test-from-episode", type=int)
    parser.add_argument("-m", "--max-episode", type=int, help="Stop running at train episode MAX_EPISODE in "
                                                              "train/test mode, or at test episode MAX_EPISODE in "
                                                              "test mode.")
    parser.add_argument("-D", "--directory", type=str)
    parser.add_argument("-p", "--port", type=int)
    parser.add_argument("-i", "--input-loadout", type=int)

    args = parser.parse_args()

    if args.test_from_episode:
        if args.load:
            exit("'load' and 'test-from-episode' cannot be both set.")
        elif args.save_period:
            exit("'save-period' and 'test-from-episode' cannot be both set.")

    return args


def launchHFO(input_data: dict, port: int, gnome_terminal: bool, visualizer: bool) -> Popen:
    gnome_terminal_command = "xterm -e " if gnome_terminal else ""

    background_process = "" if gnome_terminal else " &"

    agent_type = input_data["agent_type"]
    agent_is_binary = not is_custom(agent_type)

    teammates_type = input_data["teammates_type"]
    teammates_are_binary = not is_custom(teammates_type)

    opponents_type = input_data["opponents_type"]
    opponents_are_binary = not is_custom(opponents_type)

    offense_type = teammates_type if teammates_are_binary else agent_type
    offense_is_binary = agent_is_binary or teammates_are_binary

    if agent_is_binary and teammates_are_binary and agent_type != teammates_type:
        exit("[ERROR]: 'agent_type' and 'teammates_type' must be the same if both are binary.")

    num_offense_npcs = int(agent_is_binary) + input_data["num_teammates"] * int(teammates_are_binary)
    num_defense_npcs = input_data["num_opponents"] * int(opponents_are_binary)

    hfo_args = [
        " --no-logging",
        " --port {}".format(port) if port else "",
        " --no-sync" if visualizer else " --headless",
        " --fullstate" if input_data["fullstate"] else "",
        " --offense-agents {}".format(input_data["num_teammates"] + 1 - num_offense_npcs),
        " --offense-npcs {}".format(num_offense_npcs),
        " --offense-team {}".format(get_team_name(offense_type)) if offense_is_binary else "",
        " --defense-agents {}".format(input_data["num_opponents"] - num_defense_npcs),
        " --defense-npcs {}".format(num_defense_npcs),
        " --defense-team {}".format(get_team_name(opponents_type)) if opponents_are_binary else "",
        " --frames-per-trial {}".format(input_data["frames_per_trial"]),
        " --untouched-time {}".format(input_data["untouched_time"])
    ]

    unformatted_command = "{}../HFO/bin/HFO " + "{}" * len(hfo_args) + "{}"
    formatted_command = unformatted_command.format(gnome_terminal_command, *hfo_args, background_process)
    environment_variables = {**os.environ, "LC_ALL": "C"}

    return Popen(formatted_command, shell=True, env=environment_variables, start_new_session=True)


def launchOtherAgents(directory: str, port: int, input_loadout: int, input_data: dict,
                      wait_for_quit_thread: WaitForQuitThread) -> None:
    if is_custom(input_data["teammates_type"]):
        agent_type = input_data["agent_type"]
        team_name = "base" if is_custom(agent_type) else get_team_name(agent_type)
        for _ in range(int(input_data["num_teammates"])):
            TeammateThread(directory, port, input_loadout, input_data["teammates_type"],
                           team_name, wait_for_quit_thread).start()

    if is_custom(input_data["opponents_type"]):
        for _ in range(int(input_data["num_opponents"])):
            OpponentThread(directory, port, input_loadout, input_data["opponents_type"],
                           "base", wait_for_quit_thread).start()


def evaluateAgent(agent: LearningAgentForHFO, directory: str, args: argparse.Namespace, input_data: dict,
                  wait_for_quit_thread: Thread) -> None:
    num_episodes = {
        "test": input_data["num_test_episodes"],
        "train": input_data["num_train_episodes"],
        "total": input_data["num_test_episodes"] + input_data["num_train_episodes"],
        "save": args.save_period or 1,
        "max": args.max_episode or sys.maxsize
    }
    episode, train_episode = getEpisodeAndTrainEpisode(
        directory, args.load, args.test_from_episode, num_episodes)

    if args.load:
        loadAgent(agent, directory, -1, num_episodes)
    elif args.test_from_episode:
        loadAgent(agent, directory, train_episode, num_episodes)
    else:
        createOutputFiles(directory, agent)

    if args.test_from_episode:
        agent.setLearning(False)
        playTestEpisodes(agent, num_episodes["max"], wait_for_quit_thread)
    else:
        playEpisodes(agent, directory, episode, num_episodes, wait_for_quit_thread)


def createOutputFiles(directory: str, agent: LearningAgentForHFO) -> None:
    with open(getPath(directory, "test-output"), "w"):
        pass

    with open(getPath(directory, "train-output"), "w") as train_output_file:
        train_output_file.write("Episode\t\tAverage loss\n\n")

    save_data = {
        "next_episode": 0,
        "next_test_episode": 0,
        "current_test_rollout_goals": 0,
        "next_train_episode": 0,
        "execution_time": 0,
        "execution_time_readable": getReadableTime(0)
    }
    agent.saveParameters(save_data)

    writeTxt(getPath(directory, "save"), save_data)


def getEpisodeAndTrainEpisode(directory: str, load: bool, test_from_episode: int,
                              num_episodes: dict) -> tuple:
    if load:
        return loadEpisodeAndTrainEpisode(directory)
    elif test_from_episode:
        return test_from_episode // num_episodes["train"] * num_episodes["total"], \
               test_from_episode
    else:
        return 0, 0


def loadEpisodeAndTrainEpisode(directory: str) -> tuple:
    save_data = readTxt(getPath(directory, "save"))
    return int(save_data["next_episode"]), int(save_data["next_train_episode"])


def loadAgent(agent: LearningAgentForHFO, directory: str, train_episode: int,
              num_episodes: "dict[str, int]") -> None:
    agent_state_path = getAgentStatePath(directory, train_episode, num_episodes["train"])

    if os.path.exists(agent_state_path):
        print("[INFO] Loading agent from file:", agent_state_path)
        agent.load(agent_state_path)
    else:
        print("[INFO] Path '" + agent_state_path + "' not found. Agent not loaded.")


def playTestEpisodes(agent: AgentForHFO, max_episode: int, wait_for_quit_thread: Thread):
    episode = 0
    while wait_for_quit_thread.is_alive() and episode < max_episode and agent.playEpisode():
        print(f'Test episode {episode} ended with {hfo.STATUS_STRINGS[agent.status]}')
        episode += 1


def playEpisodes(agent: LearningAgentForHFO, directory: str, episode: int, num_episodes: dict,
                 wait_for_quit_thread: Thread) -> None:
    last_time = time.time()
    server_running = True
    saved = False
    unsaved_data = {"test": [], "train": []}

    while wait_for_quit_thread.is_alive() and server_running and not reachedMaxTrainEpisode(episode, num_episodes):
        last_time, server_running, saved = playEpisode(agent, directory, episode, num_episodes, last_time, unsaved_data)
        episode += 1

    if server_running and not saved:  # Exit by quit
        saveData(directory, agent, num_episodes, time.time() - last_time, unsaved_data)


def playEpisode(agent: LearningAgentForHFO, directory: str, episode: int, num_episodes: dict,
                last_time: float, unsaved_data: dict) -> tuple:
    is_training, episode_type, episode_type_index, rollout_episode, rollout = getEpisodeInfo(episode, num_episodes)

    if episode % num_episodes["total"] == 0:
        saveAgent(agent, directory, rollout * num_episodes["train"], num_episodes["train"])

    agent.setLearning(is_training)
    server_up = agent.playEpisode()

    print('{} episode {} ({} of rollout {}) ended with {}'.format(
        episode_type, episode_type_index, rollout_episode, rollout,
        hfo.STATUS_STRINGS[agent.status]
    ))

    if episode_type == "Train":
        unsaved_data["train"].append({
            "train_episode": episode_type_index,
            "average_loss": agent.average_loss
        })
    else:  # "Test"
        if episode_type_index % num_episodes["test"] == 0 or len(unsaved_data["test"]) == 0:  # First episode in rollout
            unsaved_data["test"].append({
                "train_episode": rollout * num_episodes["train"],
                "rollout_goals": int(agent.status == GOAL),
                "rollout_episodes": 1,
                "finished_rollout": False
            })
        else:
            unsaved_data["test"][-1]["rollout_goals"] += int(agent.status == GOAL)
            unsaved_data["test"][-1]["rollout_episodes"] += 1

        if (episode_type_index + 1) % num_episodes["test"] == 0:  # Last episode in rollout
            unsaved_data["test"][-1]["finished_rollout"] = True

    saved = False
    if (episode + 1) % num_episodes["save"] == 0:
        current_time = time.time()
        saveData(directory, agent, num_episodes, current_time - last_time, unsaved_data)
        last_time = current_time
        saved = True

    return last_time, server_up, saved


def reachedMaxTrainEpisode(episode: int, num_episodes: dict) -> int:
    is_training, _, episode_type_index, _, _ = getEpisodeInfo(episode, num_episodes)
    if is_training:
        return episode_type_index >= num_episodes["max"]
    return (episode - episode_type_index) > num_episodes["max"]


def getEpisodeInfo(episode: int, num_episodes: dict) -> tuple:
    rollout = episode // num_episodes["total"]
    rollout_episode = episode % num_episodes["total"]
    is_training = rollout_episode >= num_episodes["test"]
    if is_training:
        rollout_episode -= num_episodes["test"]

    episode_type = "Train" if is_training else "Test"
    type_multiplier = num_episodes["train"] if is_training else num_episodes["test"]
    episode_type_index = type_multiplier * rollout + rollout_episode

    return is_training, episode_type, episode_type_index, rollout_episode, rollout


def saveAgent(agent: LearningAgentForHFO, directory: str, train_episode: int = -1,
              num_train_episodes: int = 1) -> None:
    agent_state_path = getPath(directory, "agent-state")
    if not os.path.exists(agent_state_path):
        os.mkdir(agent_state_path)

    agent_state_full_path = getAgentStatePath(directory, train_episode, num_train_episodes)
    if not os.path.exists(agent_state_full_path):
        os.mkdir(agent_state_full_path)

    agent.save(agent_state_full_path)


def saveData(directory: str, agent: LearningAgentForHFO, num_episodes: dict, delta_time: float,
             unsaved_data: dict) -> None:
    train_episodes_elapsed = len(unsaved_data["train"])
    test_episodes_elapsed = sum(map(lambda data: data["rollout_episodes"], unsaved_data["test"]))

    save_path = getPath(directory, "save")
    save_data = readTxt(save_path)

    agent.saveParameters(save_data)
    if train_episodes_elapsed > 0:
        saveTrainData(directory, unsaved_data["train"])
        saveAgent(agent, directory)
    if test_episodes_elapsed > 0:
        saveTestData(directory, unsaved_data["test"], save_data, num_episodes["test"])

    save_data["next_episode"] = int(save_data["next_episode"]) + train_episodes_elapsed + test_episodes_elapsed
    save_data["next_train_episode"] = int(save_data["next_train_episode"]) + train_episodes_elapsed
    save_data["next_test_episode"] = int(save_data["next_test_episode"]) + test_episodes_elapsed
    save_data["execution_time"] = float(save_data["execution_time"]) + delta_time
    save_data["execution_time_readable"] = getReadableTime(save_data["execution_time"])

    writeTxt(save_path, save_data)


def saveTrainData(directory: str, train_data: list) -> None:
    with open(getPath(directory, "train-output"), "a") as file:
        file.writelines([
            f"{data['train_episode']}\t\t{data['average_loss']}\n"
            for data in train_data
        ])

    train_data.clear()


def saveTestData(directory: str, test_data: list, save_data: dict, num_test_episodes: int) -> None:
    rollout_goals = int(save_data["current_test_rollout_goals"])
    if rollout_goals > 0:
        test_data[0]["rollout_goals"] += rollout_goals
        save_data["current_test_rollout_goals"] = 0

    if not test_data[-1]["finished_rollout"]:
        save_data["current_test_rollout_goals"] = test_data[-1]["rollout_goals"]

    with open(getPath(directory, "test-output"), "a") as file:
        file.writelines([
            "% goals after {} train episodes: {}%\n".format(
                data["train_episode"],
                data["rollout_goals"] * 100 / num_test_episodes
            ) for data in test_data if data["finished_rollout"]
        ])

    test_data.clear()


def killProcesses(hfo_process: Popen, gnome_terminal: bool):
    if gnome_terminal:
        for p in hfo_process.children(recursive=True):
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        hfo_process.kill()
    else:
        os.killpg(os.getpgid(hfo_process.pid), SIGTERM)
        os.kill(hfo_process.pid, SIGTERM)


if __name__ == '__main__':
    main()
