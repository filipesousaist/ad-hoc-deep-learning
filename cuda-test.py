import sys

sys.path.append("../ATPO-Policy")
from agents import DRQNAgent

import gym

from yaaf import Timestep


def run(num_episodes: int):
    name = "CartPole-v0"
    env = gym.make(name)
    num_features = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent = agent = DRQNAgent(num_features, num_actions, hidden_sizes=32, cuda=True)
    for e in range(num_episodes):
        state = env.reset()
        env.render()
        terminal = False
        while not terminal:
            action = agent.action(state)
            next_state, reward, terminal, info = env.step(action)
            env.render()
            timestep = Timestep(state, action, reward, next_state, terminal, info)
            agent.reinforcement(timestep)
            state = next_state

if __name__ == '__main__':

    run(num_episodes=5)