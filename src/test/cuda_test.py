import time

from src.lib.ATPO_policy import DRQNAgent
from yaaf.agents.dqn import MLPDQNAgent

import gym

from yaaf import Timestep

from src.lib.time import getReadableTime

import argparse


def main():
    run(num_episodes=10)


def run(num_episodes: int):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", action="store_true")
    args = parser.parse_args()

    name = "MountainCar-v0"
    env = gym.make(name)
    num_features = env.observation_space.shape[0]
    num_actions = env.action_space.n
    #agent = MLPDQNAgent(num_features, num_actions, cuda=args.cuda, layers=((1024, "relu"), (1024,"relu"),(1024,"relu")))
    agent = DRQNAgent(num_features, num_actions, cuda=args.cuda, num_layers=3)
    t = time.time()
    for e in range(num_episodes):
        state = env.reset()
        #env.render()
        terminal = False
        while not terminal:
            action = agent.action(state)
            next_state, reward, terminal, info = env.step(action)
            #env.render()
            timestep = Timestep(state, action, reward, next_state, terminal, info)
            agent.reinforcement(timestep)
            state = next_state
        print("Episode {}: {}".format(e, getReadableTime(time.time() - t)))


if __name__ == '__main__':
    main()
