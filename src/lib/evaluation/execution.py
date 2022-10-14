import time

import hfo
from hfo import GOAL

from src.hfo_agents.AgentForHFO import AgentForHFO
from src.hfo_agents.learning.LearningAgentForHFO import LearningAgentForHFO

from src.lib.evaluation.episodes import reachedMaxTrainEpisode, getEpisodeInfo
from src.lib.evaluation.storage import saveData, saveAgent
from src.lib.threads import WaitForQuitThread


def playTestEpisodes(agent: AgentForHFO, max_episode: int, wait_for_quit_thread: WaitForQuitThread):
    episode = 0
    while wait_for_quit_thread.is_running() and episode < max_episode and agent.playEpisode():
        print(f'Test episode {episode} ended with {hfo.STATUS_STRINGS[agent.status]}')
        episode += 1


def playEpisodes(agent: LearningAgentForHFO, directory: str, episode: int, num_episodes: dict,
                 wait_for_quit_thread: WaitForQuitThread) -> None:
    last_time = time.time()
    server_running = True
    saved = False
    unsaved_data = {"test": [], "train": []}

    while wait_for_quit_thread.is_running() and server_running and not reachedMaxTrainEpisode(episode, num_episodes):
        last_time, server_running, saved = _playEpisode(agent, directory, episode, num_episodes, last_time, unsaved_data)
        episode += 1

    if server_running and not saved:  # Exit by quit or by max train episode reached
        saveData(directory, agent, num_episodes, time.time() - last_time, unsaved_data)


def _playEpisode(agent: LearningAgentForHFO, directory: str, episode: int, num_episodes: dict,
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
