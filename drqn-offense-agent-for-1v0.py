#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1


from math import sqrt
import sys, os
import argparse

from hfo import *

sys.path.append("../ATPO-Policy")
from agents import DRQNAgent
from yaaf.agents.dqn import MLPDQNAgent
from yaaf import Timestep

from txtLib import *

GOAL_COORDS = [1, 0]
MAX_DISTANCE = 2 * sqrt(2)
VALIDATION_FEATURES = {SHOOT: (5, 1), DRIBBLE: (5, 1), GO_TO_BALL: (5, -1), REORIENT: (5, -1)}
ACTIONS = [SHOOT, DRIBBLE, GO_TO_BALL, REORIENT]
NUM_TRAIN_EPISODES = 100
NUM_TEST_EPISODES = 30

FILE_NAME = os.path.basename(__file__).rstrip(".py")
DEFAULT_TEST_OUTPUT_PATH = "./output/" + FILE_NAME + "-TEST-results.txt"
DEFAULT_TRAIN_OUTPUT_PATH = "./output/" + FILE_NAME + "-TRAIN-results.txt"
DEFAULT_AGENT_STATE_PATH = "./agent-state"
DEFAULT_SAVE_PATH = "./save.txt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load-path", type=str)
    parser.add_argument("-a", "--agent-state-path", type=str)
    parser.add_argument("-t", "--test-output-path", type=str)
    parser.add_argument("-T", "--train-output-path", type=str)
    args = parser.parse_args()

    if args.load_path and (args.agent_state_path or args.test_output_path or args.train_output_path):
        print("Invalid arguments: agent-state-file, test-output-file and train-output-file cannot be used if load-file is provided")
        sys.exit(1)

    # Create the HFO Environment
    hfo = HFOEnvironment()
    # Connect to the server with the specified
    # feature set. See feature sets in hfo.py/hfo.hpp.
    hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                        '../HFO/bin/teams/base/config/formations-dt', 6000,
                        'localhost', 'base_left', False)

    num_features = 12 + 3 * hfo.getNumOpponents() + 6 * hfo.getNumTeammates() #+ 1
    #agent = DRQNAgent(num_features, len(ACTIONS), learning_rate=0.00025,
    #    initial_exploration_rate=1, final_exploration_rate=0.1,
    #    discount_factor=0.99, final_exploration_step=1000000,
    #    target_network_update_frequency=75, num_layers=2)
    agent = MLPDQNAgent(num_features, len(ACTIONS), learning_rate=0.00025,
        initial_exploration_rate=1, final_exploration_rate=0.1, final_exploration_step=1000000,
        discount_factor=0.99, target_network_update_frequency=75)
    
    episode, agent_state_path, test_output_file, train_output_file, resume = \
        loadOutputFiles(args.load_path) if args.load_path else \
        createOutputFiles(args.agent_state_path, args.test_output_path, args.train_output_path)
    
    if args.load_path:
        agent_state_full_path = agent_state_path + "/after{}episodes".format(episode)
        if os.path.exists(agent_state_full_path):
            print("[INFO] Loading agent from file:", agent_state_full_path)
            agent.load(agent_state_full_path)
        else:
            print("[INFO] Path", agent_state_full_path, "not found. Agent not loaded.")

    resume = False

    while True:
        if episode % NUM_TRAIN_EPISODES == 0 and not resume: # Test
            runTestPhase(episode, hfo, agent, test_output_file)
            train_output_file.flush()
            saveProgress(agent, episode)

        resume = False

        result = playEpisode(episode, hfo, agent, learn=True)
        train_output_file.write("{}\t\t{}\n".format(episode, result["average_loss"]))

        episode += 1


def createOutputFiles(agent_state_path, test_output_path, train_output_path):   
    if not agent_state_path:
        agent_state_path = DEFAULT_AGENT_STATE_PATH
    if not test_output_path:
        test_output_path = DEFAULT_TEST_OUTPUT_PATH
    if not train_output_path:
        train_output_path = DEFAULT_TRAIN_OUTPUT_PATH

    if not os.path.exists(agent_state_path):
        os.mkdir(agent_state_path)

    test_output_file = open(test_output_path, "w")
    test_output_file.write("NUM_TEST_EPISODES = {}\n\n".format(NUM_TEST_EPISODES))
    test_output_file.flush()

    train_output_file = open(train_output_path, "w")
    train_output_file.writelines([
        "NUM_TRAIN_EPISODES = {}\n\n".format(NUM_TRAIN_EPISODES),
        "Episode\t\tAverage loss\n\n"
    ])
    train_output_file.flush()

    writeTxt(DEFAULT_SAVE_PATH, {
        "agent_state_path": agent_state_path,
        "test_output_path": test_output_path,
        "train_output_path": train_output_path,
        "current_train_episode": "0"
    })

    return 0, agent_state_path, test_output_file, train_output_file, False


def loadOutputFiles(save_path):
    loadedData = readTxt(save_path)

    test_output_file = open(loadedData["test_output_path"], "a")
    train_output_file = open(loadedData["train_output_path"], "a")

    return int(loadedData["current_train_episode"]), loadedData["agent_state_path"], \
        test_output_file, train_output_file, True


def saveProgress(agent, current_train_episode):
    saveData = readTxt(DEFAULT_SAVE_PATH)

    # Save agent state
    dirName = saveData["agent_state_path"] + "/after{}episodes".format(current_train_episode)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    agent.save(dirName)

    # Save current episode to file
    saveData["current_train_episode"] = str(current_train_episode)
    writeTxt(DEFAULT_SAVE_PATH, saveData)


def playEpisode(episode, hfo, agent, learn=True):
    # Maybe there is a better way to reset hidden state?
    agent._last_hidden = None

    episode_loss = 0
    num_timesteps = 0

    status = IN_GAME
    observation = hfo.getState()
    #observation = np.append(observation, -1.0)

    info = None

    while status == IN_GAME:
        num_timesteps += 1

        action = agent.action(observation)

        hfo_action = ACTIONS[action]
        
        hfo.act(hfo_action if isActionValid(hfo_action, observation) else NOOP)
        
        status = hfo.step()
        next_observation = hfo.getState()
        #next_observation = np.append(next_observation, (num_timesteps - 100)/100)
        #print(next_observation)

        reward =     0 if status == IN_GAME \
             else  100 if status == GOAL \
             else -100

        timestep = Timestep(observation, action, reward, next_observation, status != IN_GAME, {}) 

        info = agent.reinforcement(timestep)
        
        if learn and "Loss" in info:
            episode_loss += info["Loss"]      
        
        observation = next_observation
    
    print(info)

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

def runTestPhase(current_train_episode, hfo, agent, output_file): 
    agent.eval()

    numGoals = 0
    for testEpisode in range(NUM_TEST_EPISODES):
        numGoals += int(playEpisode(testEpisode, hfo, agent, learn=False)["goal"])
    output_file.write("% goals after {} train episodes: {}%\n".format(current_train_episode, numGoals * 100 / NUM_TEST_EPISODES))
    output_file.flush()

    agent.train()

def selectActionHandCoded(observation):
    if int(observation[5]) == 1:
        if distanceToGoal(observation) <= 0.4:
            return SHOOT
        else:
            return DRIBBLE
    return GO_TO_BALL

def isActionValid(action, observation):
    if action not in VALIDATION_FEATURES:
        return True
    feature, value = VALIDATION_FEATURES[action]
    return int(observation[feature]) == value


def distanceToGoal(observation):
    return sqrt((observation[3] - GOAL_COORDS[0])**2 + (observation[4] - GOAL_COORDS[1])**2) / MAX_DISTANCE


if __name__ == '__main__':
    main()
