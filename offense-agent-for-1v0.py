#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1


from math import sqrt
import sys, os
import argparse
import time

from hfo import *

sys.path.append("../ATPO-Policy")
from agents import DRQNAgent
from yaaf.agents.dqn import MLPDQNAgent
from yaaf import Timestep

sys.path.append("./lib")
from txtLib import *
from timeLib import *
from constants import *

GOAL_COORDS = [1, 0]
MAX_DISTANCE = 2 * sqrt(2)
VALIDATION_FEATURES = {
    SHOOT: (5, 1),
    DRIBBLE: (5, 1),
    PASS: (5, 1),
    MOVE: (5, -1),
    GO_TO_BALL: (5, -1),
    REORIENT: (5, -1)}
ACTIONS = [SHOOT, DRIBBLE, PASS, MOVE, REORIENT] # GO_TO_BALL
NUM_TRAIN_EPISODES = 100
NUM_TEST_EPISODES = 30

SAVE_FILE_NAME = "save.txt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", choices=["dqn", "drqn"])
    parser.add_argument("-l", "--load", action="store_true")
    parser.add_argument("-e", "--train-episode", type=int)
    parser.add_argument("-t", "--test-only", action="store_true")
    parser.add_argument("-c", "--custom-features", action="store_true")
    parser.add_argument("-o", "--output-path", type=str)
    args = parser.parse_args()

    if args.train_episode and not args.load:
        sys.exit("[ERROR] Train episode can only be specified when loading agent from file.")

    # Create the HFO Environment
    hfo = HFOEnvironment()
    # Connect to the server with the specified
    # feature set. See feature sets in hfo.py/hfo.hpp.
    hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                        '../HFO/bin/teams/base/config/formations-dt', 6000,
                        'localhost', 'base_left', False)

    num_opponents = hfo.getNumOpponents()
    num_teammates = hfo.getNumTeammates()

    num_features = 12 + 3 * num_opponents + 6 * num_teammates
    if args.custom_features:
        num_features = len(extractFeatures(np.zeros(num_features)))

    hfo_info = {
        "num_oppnonents": num_opponents,
        "num_teammates": num_teammates,
        "num_features": num_features,
        "num_actions": len(ACTIONS),
        "actions": [ACTION_STRINGS[action] for action in ACTIONS]
    }

    
    agent = MLPDQNAgent(hfo_info["num_features"], hfo_info["num_actions"], learning_rate=0.00025,
        initial_exploration_rate=1, final_exploration_rate=0.1, final_exploration_step=5000000,
        discount_factor=0.99, target_network_update_frequency=75) \
            \
            if args.agent == "dqn" else \
            \
            DRQNAgent(hfo_info["num_features"], hfo_info["num_actions"], learning_rate=0.001,
        initial_exploration_rate=1, final_exploration_rate=0.1,
        discount_factor=0.99, final_exploration_step=500000,
        target_network_update_frequency=75, num_layers=2, cuda=False)
    
    output_path = args.output_path or DEFAULT_OUTPUT_PATH
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    print("[INFO] Output path: " + output_path)
    
    file_paths = getFilePaths(output_path) 

    test_output_file, train_output_file, episode = \
        loadOutputFiles(file_paths, agent, args.train_episode) if args.load else \
        createOutputFiles(file_paths, hfo_info) if not args.test_only else \
        (None, None, 0)

    if args.test_only:
        agent.eval()
        while True:
            playEpisode(episode, hfo, agent, args.custom_features, learn=False)
            episode += 1
    else:
        resume = args.load

        last_time = time.time()
        while True:
            if episode % NUM_TRAIN_EPISODES == 0 and not resume and NUM_TEST_EPISODES > 0: # Test
                runTestPhase(episode, hfo, agent, test_output_file, args.custom_features)
                train_output_file.flush()

                current_time = time.time()
                saveProgress(file_paths, agent, episode, current_time - last_time)
                last_time = current_time

            resume = False

            result = playEpisode(episode, hfo, agent, args.custom_features, learn=True)
            train_output_file.write("{}\t\t{}\n".format(episode, result["average_loss"]))

            episode += 1


def getFilePaths(output_path):
    return {
        "train-output" : output_path + "/" + TRAIN_OUTPUT_FILE_NAME,
        "test-output": output_path + "/" + TEST_OUTPUT_FILE_NAME,
        "agent-state": output_path + "/" + AGENT_STATE_FILE_NAME,
        "save": output_path + "/" + SAVE_FILE_NAME
    }


def createOutputFiles(file_paths, hfo_info):
    if not os.path.exists(file_paths["agent-state"]):
        os.mkdir(file_paths["agent-state"])
    
    test_output_file = open(file_paths["test-output"], "w")
    test_output_file.write("NUM_TEST_EPISODES = {}\n\n".format(NUM_TEST_EPISODES))
    test_output_file.flush()

    train_output_file = open(file_paths["train-output"], "w")
    train_output_file.writelines([
        "NUM_TRAIN_EPISODES = {}\n\n".format(NUM_TRAIN_EPISODES),
        "Episode\t\tAverage loss\n\n"
    ])
    train_output_file.flush()

    writeTxt(file_paths["save"], {
        "current_train_episode": 0,
        "execution_time": 0,
        "execution_time_readable": getReadableTime(0),
        **hfo_info
    })

    return test_output_file, train_output_file, 0


def loadOutputFiles(file_paths, agent, train_episode):
    loaded_data = readTxt(file_paths["save"])

    episode = train_episode or int(loaded_data["current_train_episode"])

    loadAgent(agent, file_paths["agent-state"], episode)

    test_output_file = open(file_paths["test-output"], "a")
    train_output_file = open(file_paths["train-output"], "a")

    return test_output_file, train_output_file, episode


def loadAgent(agent, base_path, episode):
    agent_state_full_path = base_path + "/after{}episodes".format(episode)
    
    if os.path.exists(agent_state_full_path):
        print("[INFO] Loading agent from file:", agent_state_full_path)
        agent.load(agent_state_full_path)
    else:
        print("[INFO] Path", agent_state_full_path, "not found. Agent not loaded.")


def saveProgress(file_paths, agent, current_train_episode, elapsed_time):
    # Save agent state
    agent_state_full_path = file_paths["agent-state"] + \
        "/after{}episodes".format(current_train_episode)
    if not os.path.exists(agent_state_full_path):
        os.mkdir(agent_state_full_path)
    agent.save(agent_state_full_path)

    save_data = readTxt(file_paths["save"])
    # Save current episode to file
    save_data["current_train_episode"] = current_train_episode
    save_data["execution_time"] = float(save_data["execution_time"]) + elapsed_time
    save_data["execution_time_readable"] = getReadableTime(save_data["execution_time"])
    writeTxt(file_paths["save"], save_data)


def playEpisode(episode, hfo, agent, custom_features=False, learn=True): 
    # Maybe there is a better way to reset hidden state?
    agent._last_hidden = None

    episode_loss = 0
    num_timesteps = 0
    
    info = None

    status = IN_GAME
    observation = hfo.getState()
    features = extractFeatures(observation) if custom_features else observation

    while status == IN_GAME:
        num_timesteps += 1

        action = agent.action(features)

        hfo_action = ACTIONS[action] if isActionValid(ACTIONS[action], observation) else NOOP
        
        if hfo_action == PASS:
            hfo.act(PASS, 7)
        else:
            hfo.act(hfo_action)
        
        status = hfo.step()
        next_observation = hfo.getState()
        next_features = extractFeatures(next_observation) if custom_features else next_observation

        reward =    0 if status == IN_GAME \
            else  100 if status == GOAL \
            else -100

        timestep = Timestep(features, action, reward, next_features, status != IN_GAME, {}) 

        info = agent.reinforcement(timestep)
        
        if learn and "Loss" in info:
            episode_loss += info["Loss"]      
        
        observation = next_observation
        features = next_features
    
    print(info)

    episode_type = "Train" if learn else "Test"
    # Check the outcome of the episode
    print(('%s episode %d ended with %s' % (episode_type, episode, hfo.statusToString(status))))

    # Quit if the server goes down
    if status == SERVER_DOWN:
        hfo.act(QUIT)
        exit()

    return {
        "goal": status == GOAL,
        "average_loss": episode_loss / num_timesteps if num_timesteps > 0 else None
    }

def runTestPhase(current_train_episode, hfo, agent, output_file, custom_features=False): 
    agent.eval()

    num_goals = 0
    for test_episode in range(NUM_TEST_EPISODES):
        num_goals += int(playEpisode(test_episode, hfo, agent, custom_features, learn=False)["goal"])
    output_file.write("% goals after {} train episodes: {}%\n".format(current_train_episode, num_goals * 100 / NUM_TEST_EPISODES))
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


def extractFeatures(observation):
    # New features
    # 0 Able to Kick
    # 1 Goal center proximity
    # 2 Goal center angle
    # 3 Distance from agent to nearest edge
    # 4 Distance from ball to nearest edge

    return np.array([
        *observation[5:8],
        distanceToNearestEdge(*observation[0:2]),
        distanceToNearestEdge(*observation[3:5])    
    ])


def isDangerousToShoot(observation):
    ball_to_edge_distance = distanceToNearestEdge(*observation[3:5])
    return ball_to_edge_distance < 0.05 and ball_to_edge_distance < distanceToNearestEdge(*observation[0:2]) 

def distanceToNearestEdge(x, y):
    return min(1 - abs(x), 1 - abs(y))

if __name__ == '__main__':
    main()