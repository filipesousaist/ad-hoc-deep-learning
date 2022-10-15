import time

from src.lib.agents import DRQNAgent
from yaaf.agents.dqn import MLPDQNAgent

import gym

from yaaf import Timestep

from src.lib.time import getReadableTime

import argparse


def main():
    for hs in (50, 100, 150):
        print(f"hs={hs}")
        run(num_train_episodes=50, num_test_episodes=50, hidden_sizes=hs)


def run(num_train_episodes: int, num_test_episodes: int, hidden_sizes: int):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", action="store_true")
    args = parser.parse_args()

    name = "CartPole-v1"
    env = gym.make(name)
    num_features = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent = MLPDQNAgent(num_features, num_actions, cuda=args.cuda, layers=((hidden_sizes, "relu"), (hidden_sizes,"relu"),(hidden_sizes,"relu")))
    #agent = DRQNAgent(num_features, num_actions, cuda=args.cuda, num_layers=2, hidden_sizes=hidden_sizes,
    #                  target_network_update_frequency=50, trajectory_update_length=10)

    num_episodes = 0
    for mode in ("train", "test"):
        if mode == "train":
            agent.train()
            num_episodes = num_train_episodes
        else:
            agent.eval()
            num_episodes = num_test_episodes
        print(mode)
        t = time.time()
        total_total_reward = 0
        for e in range(num_episodes):
            state = env.reset()
            #env.render()
            terminal = False
            total_reward = 0
            steps = 0
            while not terminal:
                steps += 1
                action = agent.action(state)
                next_state, reward, terminal, info = env.step(action)
                total_reward += reward
                #env.render()
                timestep = Timestep(state, action, reward, next_state, terminal, info)
                agent.reinforcement(timestep)
                state = next_state
            print("Episode {}: {}, reward={}".format(e, getReadableTime(time.time() - t), total_reward))
            total_total_reward += total_reward
        print(f"Total reward={total_total_reward}")


if __name__ == '__main__':
    main()
