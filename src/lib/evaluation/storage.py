import os

from src.hfo_agents.learning.LearningAgentForHFO import LearningAgentForHFO
from src.lib.paths import getPath, getAgentStatePath
from src.lib.io import readTxt, writeTxt
from src.lib.time import getReadableTime


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


def loadAgent(agent: LearningAgentForHFO, directory: str, train_episode: int, num_train_episodes: int) -> None:
    agent_state_path = getAgentStatePath(directory, train_episode, num_train_episodes)

    if os.path.exists(agent_state_path):
        print("[INFO] Loading agent from file:", agent_state_path)
        agent.load(agent_state_path)
    else:
        print("[INFO] Path '" + agent_state_path + "' not found. Agent not loaded.")


def saveAgent(agent: LearningAgentForHFO, directory: str, train_episode: int = -1, num_train_episodes: int = 1) -> None:
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
        _saveTrainData(directory, unsaved_data["train"])
        saveAgent(agent, directory)
    if test_episodes_elapsed > 0:
        _saveTestData(directory, unsaved_data["test"], save_data, num_episodes["test"])

    save_data["next_episode"] = int(save_data["next_episode"]) + train_episodes_elapsed + test_episodes_elapsed
    save_data["next_train_episode"] = int(save_data["next_train_episode"]) + train_episodes_elapsed
    save_data["next_test_episode"] = int(save_data["next_test_episode"]) + test_episodes_elapsed
    save_data["execution_time"] = float(save_data["execution_time"]) + delta_time
    save_data["execution_time_readable"] = getReadableTime(save_data["execution_time"])

    writeTxt(save_path, save_data)


def _saveTrainData(directory: str, train_data: list) -> None:
    with open(getPath(directory, "train-output"), "a") as file:
        file.writelines([
            f"{data['train_episode']}\t\t{data['average_loss']}\n"
            for data in train_data
        ])

    train_data.clear()


def _saveTestData(directory: str, test_data: list, save_data: dict, num_test_episodes: int) -> None:
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
