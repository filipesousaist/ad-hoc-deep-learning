import argparse
import os
from threading import Thread
import time
from typing import Type

import hfo
from hfo import GOAL

from src.lib.paths import DEFAULT_DIRECTORY, DEFAULT_PORT, getPath
from src.lib.io import logOutput, flushOutput, readTxt, writeTxt
from src.lib.time import getReadableTime
from src.lib.threads import WaitForQuitThread, TeammateThread, OpponentThread
from src.lib.input import readInputData

from src.hfo_agents.agentForHFOFactory import getAgentForHFOFactory
from src.hfo_agents.LearningAgentForHFO import LearningAgentForHFO


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

    launchHFO(input_data, port, args.gnome_terminal, args.visualizer)
    time.sleep(2)
    launchOtherAgents(directory, port, input_loadout, input_data, wait_for_quit_thread)

    agent_type = input_data["agent_type"]
    if agent_type != "npc":
        agent = getAgentForHFOFactory(agent_type)(directory, port, "base_left", input_loadout)
        evaluateAgent(agent, directory, args, input_data, wait_for_quit_thread)
    else:
        while wait_for_quit_thread.is_alive():
            pass

    print("[INFO] Terminating 'evaluate.py' and associated processes...")

    if not args.no_output:
        flushOutput(getPath(directory, "output"))

    os.system("killall rcssserver -9")
    os.system("killall python")


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualizer", action="store_true", help="launch HFO visualizer")
    parser.add_argument("-g", "--gnome-terminal", action="store_true", help="lauch agent in an external terminal")
    parser.add_argument("-n", "--no-output", action="store_true")
    
    parser.add_argument("-l", "--load", action="store_true", help="load data stored in save file")
    parser.add_argument("-t", "--test-from-episode", type=int)
    parser.add_argument("-D", "--directory", type=str)
    parser.add_argument("-p", "--port", type=int)
    parser.add_argument("-i", "--input-loadout", type=int)

    args = parser.parse_args()
    
    if args.load and args.test_from_episode:
        exit("'load' and 'test-from-episode' cannot be both set.")

    return args


def launchHFO(input_data: dict, port: int, gnome_terminal: bool, visualizer: bool) -> None:
    gnome_terminal_command = "gnome-terminal -- " if gnome_terminal else ""
    background_process = "" if gnome_terminal else " &" 

    num_offense_agents = int(input_data["agent_type"] != "npc") + \
        input_data["num_teammates"] * int(input_data["teammates_type"] != "npc")
    num_defense_agents = input_data["num_opponents"] * int(input_data["opponents_type"] != "npc")
    
    hfo_args = [
        " --port {}".format(port) if port else "",
        " --no-sync" if visualizer else " --headless",
        " --fullstate" if input_data["fullstate"] else "",
        " --offense-agents {}".format(num_offense_agents),
        " --offense-npcs {}".format(input_data["num_teammates"] + 1 - num_offense_agents),
        " --defense-agents {}".format(num_defense_agents),
        " --defense-npcs {}".format(input_data["num_opponents"] - num_defense_agents),
        " --frames-per-trial {}".format(input_data["frames_per_trial"]),
        " --untouched-time {}".format(input_data["untouched_time"]) 
    ]

    unformatted_command = "{}script -c 'LC_ALL=C ../HFO/bin/HFO " + "{}" * len(hfo_args) + "'{}"
    os.system(unformatted_command.format(gnome_terminal_command, *hfo_args, background_process))


def launchOtherAgents(directory: str, port: int, input_loadout: int, input_data: dict, wait_for_quit_thread: WaitForQuitThread) -> None:
    if input_data["teammates_type"] != "npc":
        for _ in range(int(input_data["num_teammates"])):
            TeammateThread(directory, port, input_loadout, input_data["teammates_type"], wait_for_quit_thread).start()

    if input_data["opponents_type"] != "npc":
        for _ in range(int(input_data["num_opponents"])):
            OpponentThread(directory, port, input_loadout, input_data["opponents_type"], wait_for_quit_thread).start()


def evaluateAgent(agent: Type[LearningAgentForHFO], directory: str, args: argparse.Namespace, input_data: dict,
        wait_for_quit_thread: Thread) -> None:
    num_episodes = {
        "test": input_data["num_test_episodes"],
        "train": input_data["num_train_episodes"],
        "total": input_data["num_test_episodes"] + input_data["num_train_episodes"]
    }
    episode, train_episode = getEpisodeAndTrainEpisode(
        directory, args.load, args.test_from_episode, num_episodes)
    
    if args.load or args.test_from_episode:
        loadAgent(agent, directory, train_episode, num_episodes)
    else:
        createOutputFiles(directory)
    
    if args.test_from_episode:
        playTestEpisodes(agent, wait_for_quit_thread)
    else:    
        playEpisodes(agent, directory, episode, num_episodes, wait_for_quit_thread)


def createOutputFiles(directory: str) -> None:
    with open(getPath(directory, "test-output"), "w"):
        pass

    with open(getPath(directory, "train-output"), "w") as train_output_file:
        train_output_file.write("Episode\t\tAverage loss\n\n")

    writeTxt(getPath(directory, "save"), {
        "next_episode": 0,
        "next_test_episode": 0,
        "current_test_rollout_goals": 0,
        "next_train_episode": 0,
        "execution_time": 0,
        "execution_time_readable": getReadableTime(0)
    })


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


def loadAgent(agent: Type[LearningAgentForHFO], directory: str, train_episode: int,
        num_episodes: "dict[str, int]") -> None:
    agent_state_path = getPath(directory, "agent-state") + "/after{}episodes".format(
        train_episode // num_episodes["train"] * num_episodes["train"]
    )
    if os.path.exists(agent_state_path):
        print("[INFO] Loading agent from file:", agent_state_path)
        agent.loadNetwork(agent_state_path)
    else:
        print("[INFO] Path '" + agent_state_path + "' not found. Agent not loaded.")


def playTestEpisodes(agent: Type[LearningAgentForHFO], wait_for_quit_thread: Thread):
    episode = 0
    agent.setLearning(False)
    while wait_for_quit_thread.is_alive() and agent.playEpisode():
        print(f'Test episode {episode} ended with {hfo.STATUS_STRINGS[agent.status]}')
        episode += 1


def playEpisodes(agent: Type[LearningAgentForHFO], directory: str, episode: int, num_episodes: dict,
        wait_for_quit_thread: Thread) -> None:
    server_running = True
    last_time = time.time()
    while wait_for_quit_thread.is_alive() and server_running:
        last_time, server_running = playEpisode(agent, directory, episode, num_episodes, last_time)
        episode += 1


def playEpisode(agent: Type[LearningAgentForHFO], directory: str, episode: int, num_episodes: dict,
        last_time: float) -> float:
    is_training, episode_type, episode_type_index, rollout_episode, rollout = \
        getEpisodeInfo(episode, num_episodes)

    if episode % num_episodes["total"] == 0:
        saveAgent(agent, directory, rollout * num_episodes["train"])
    
    agent.setLearning(is_training)
    server_up = agent.playEpisode()
    current_time = time.time()
    
    print('{} episode {} ({} of rollout {}) ended with {}'.format(
        episode_type, episode_type_index, rollout_episode, rollout,
        hfo.STATUS_STRINGS[agent.status]
    ))

    saveData(directory, agent, is_training, episode, num_episodes, episode_type_index,
        rollout, current_time - last_time)

    return current_time, server_up


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


def saveAgent(agent: Type[LearningAgentForHFO], directory: str, train_episode: int) -> None:
    agent_state_path = getPath(directory, "agent-state")
    if not os.path.exists(agent_state_path):
        os.mkdir(agent_state_path)
    
    agent_state_full_path = f"{agent_state_path}/after{train_episode}episodes"
    if not os.path.exists(agent_state_full_path):
        os.mkdir(agent_state_full_path)
    
    agent.saveNetwork(agent_state_full_path)


def saveData(directory: str, agent: Type[LearningAgentForHFO], is_training: bool, episode: int, 
        num_episodes: dict, episode_type_index: int, rollout: int, delta_time: float) -> None:
    save_path = getPath(directory, "save")
    
    save_data = readTxt(save_path)
    save_data["next_episode"] = episode + 1
    save_data["execution_time"] = float(save_data["execution_time"]) + delta_time
    save_data["execution_time_readable"] = getReadableTime(save_data["execution_time"])
    
    if is_training:
        saveTrainData(save_data, directory, episode_type_index, agent.average_loss)
    else:
        saveTestData(save_data, directory, num_episodes, episode_type_index, rollout, agent.status)

    writeTxt(save_path, save_data)


def saveTrainData(save_data: dict, directory: str, train_episode: int,
        average_loss: float) -> None:
    save_data["current_test_rollout_goals"] = 0
    save_data["next_train_episode"] = train_episode + 1
    
    with open(getPath(directory, "train-output"), "a") as file:
        file.write("{}\t\t{}\n".format(train_episode, average_loss))


def saveTestData(save_data: dict, directory: str, num_episodes: dict, test_episode: int,
        rollout: int, status: int) -> None:
    save_data["current_test_rollout_goals"] = \
        int(save_data["current_test_rollout_goals"]) + int(status == GOAL)
    save_data["next_test_episode"] = test_episode + 1
    
    if test_episode % num_episodes["test"] == num_episodes["test"] - 1:
        with open(getPath(directory, "test-output"), "a") as file:
            file.write("% goals after {} train episodes: {}%\n".format(
                rollout * num_episodes["train"],
                save_data["current_test_rollout_goals"] * 100 / num_episodes["test"]
            ))


if __name__ == '__main__':
    main()
