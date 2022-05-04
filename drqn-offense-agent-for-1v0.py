#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

from math import sqrt
import sys, itertools
from hfo import *
sys.path.append("../ATPO-Policy")
from agents import DRQNAgent
from yaaf import Timestep

GOAL_COORDS = [1, 0]
MAX_DISTANCE = 2 * sqrt(2)
VALIDATION_FEATURES = {SHOOT: 5, DRIBBLE: 5, MOVE: None}
ACTIONS = [SHOOT, DRIBBLE, MOVE]
NUM_TRAIN_EPISODES = 100
NUM_TEST_EPISODES = 30

def main():
    # Create the HFO Environment
    hfo = HFOEnvironment()
    # Connect to the server with the specified
    # feature set. See feature sets in hfo.py/hfo.hpp.
    hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                        '../HFO/bin/teams/base/config/formations-dt', 6000,
                        'localhost', 'base_left', False)

    num_features = 12 + 3 * hfo.getNumOpponents() + 6 * hfo.getNumTeammates()
    agent = DRQNAgent(num_features, len(ACTIONS), learning_rate=0.001, initial_exploration_rate=0.8, final_exploration_rate=0.2, num_layers=2)

    testOutputFile = open("output/drqn-offense-agent-for-1v0-TEST-results.txt", "w")
    testOutputFile.write("NUM_TEST_EPISODES = {}\n\n".format(NUM_TEST_EPISODES))
    testOutputFile.flush()

    trainOutputFile = open("output/drqn-offense-agent-for-1v0-TRAIN-results.txt", "w")
    trainOutputFile.writelines([
        "NUM_TRAIN_EPISODES = {}\n\n".format(NUM_TRAIN_EPISODES),
        "Episode\t\tAverage loss\n\n"
    ])
    trainOutputFile.flush()

    for episode in itertools.count():
        if episode % NUM_TRAIN_EPISODES == 0: # Test
            runTestPhase(episode, hfo, agent, testOutputFile)
            trainOutputFile.flush()

        result = playEpisode(episode, hfo, agent, learn=True)
        trainOutputFile.write("{}\t\t{}\n".format(episode, result["average_loss"]))


def playEpisode(episode, hfo, agent, learn=True):
    # Maybe there is a better way to reset hidden state?
    agent._last_hidden = None

    episode_loss = 0
    num_timesteps = 0

    observation = hfo.getState()
    status = hfo.step()
    while status == IN_GAME:
        action = agent.action(observation)

        hfo_action = ACTIONS[action]
        
        hfo.act(hfo_action if isActionValid(hfo_action, observation) else NOOP)
        
        next_observation = hfo.getState()
        status = hfo.step()

        reward =      0 if status == IN_GAME \
             else  1000 if status == GOAL \
             else -1000

        timestep = Timestep(observation, action, reward, next_observation, status != IN_GAME, {}) 

        info = agent.reinforcement(timestep)
        if learn and "Loss" in info:
            num_timesteps += 1
            episode_loss += info["Loss"]
        
        observation = next_observation
    
    episodeType = "Train" if learn else "Test"
    # Check the outcome of the episode
    print(('%s episode %d ended with %s' % (episodeType, episode, hfo.statusToString(status))))

    # Quit if the server goes down
    if status == SERVER_DOWN:
        hfo.act(QUIT)
        exit()

    return {
        "goal": status == GOAL,
        "average_loss": episode_loss / num_timesteps if num_timesteps > 0 else None
    }


def runTestPhase(currentTrainEpisode, hfo, agent, outputFile):
    dirName = "./agent-state/after{}episodes".format(currentTrainEpisode)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    agent.save(dirName)
    
    agent.eval()

    numGoals = 0
    for testEpisode in range(NUM_TEST_EPISODES):
        numGoals += int(playEpisode(testEpisode, hfo, agent, learn=False)["goal"])
    outputFile.write("% goals after {} train episodes: {}%\n".format(currentTrainEpisode, numGoals * 100 / NUM_TEST_EPISODES))
    outputFile.flush()

    agent.train()


def isActionValid(action, observation):
    feature = VALIDATION_FEATURES[action] 
    return feature == None or observation[feature] == 1


def distanceToGoal(observation):
    return sqrt((observation[3] - GOAL_COORDS[0])**2 + (observation[4] - GOAL_COORDS[1])**2) / MAX_DISTANCE


if __name__ == '__main__':
    main()
