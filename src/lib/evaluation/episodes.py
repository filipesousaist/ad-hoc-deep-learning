import numpy as np

from src.lib.io import readTxt, readScoreRate
from src.lib.paths import getPath


def getEpisodeAndTrainEpisode(directory: str, load: bool, test_from_episode: int,
                              num_episodes: dict) -> tuple:
    if load:
        return _loadEpisodeAndTrainEpisode(directory)
    elif test_from_episode:
        return test_from_episode // num_episodes["train"] * num_episodes["total"], \
               test_from_episode
    else:
        return 0, 0


def _loadEpisodeAndTrainEpisode(directory: str) -> tuple:
    save_data = readTxt(getPath(directory, "save"))
    return int(save_data["next_episode"]), int(save_data["next_train_episode"])


def getBestTrainEpisode(directory: str, last: int = -1) -> int:
    episodes, score_rates = readScoreRate(directory)
    if last < 0:
        return episodes[np.argmax(score_rates[0])]
    return episodes[-last:][np.argmax(score_rates[0][-last:])]


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
