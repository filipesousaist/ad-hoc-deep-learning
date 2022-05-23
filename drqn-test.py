import sys

sys.path.append("../ATPO-Policy")
from agents import DRQNAgent

import random
import numpy as np

from yaaf import Timestep

class CoinFlip:
    def __init__(self, num_guesses):
        self.num_features = 1
        self.num_actions = 2
        self._num_guesses = num_guesses

    def reset(self):
        self._guesses_done = -1 # First guess is ignored
        self._last_flip = random.randint(0, 1)
        self._flip_before_last = None

        return np.array(self._last_flip)

    def step(self, action):
        areEqual = int(self._flip_before_last == self._last_flip)

        reward = 0 if self._flip_before_last == None else \
                 1 if action == areEqual else \
                -1
        
        self._flip_before_last = self._last_flip
        self._last_flip = random.randint(0, 1)
        
        self._guesses_done += 1

        return np.array(self._last_flip), reward, self._guesses_done >= self._num_guesses, {}


def run(agent: DRQNAgent, env: CoinFlip, num_episodes: int = 1):
    total_correct_guesses = 0
    total_time = 0

    for e in range(num_episodes):
        state = env.reset()
        agent._last_hidden = None

        terminal = False
        correct_guesses = 0
        time = -1 # First step is discarded

        while not terminal:
            action = agent.action(state)
            next_state, reward, terminal, info = env.step(action)
            timestep = Timestep(state, action, reward, next_state, terminal, info)
            agent.reinforcement(timestep)
            state = next_state

            time += 1
            correct_guesses += int(reward > 0)

        print("Episode {}: {} correct guesses out of {}.".format(e, correct_guesses, time))
        total_correct_guesses += correct_guesses
        total_time += time

    return total_correct_guesses / total_time
    

if __name__ == '__main__':
    env = CoinFlip(200)
    agent = DRQNAgent(env.num_features, env.num_actions, hidden_sizes=4, target_network_update_frequency=10)

    print("Train")
    agent.train()
    print("Accuracy: {}".format(run(agent, env, num_episodes=20)))

    print("Test")
    agent.eval()
    print("Accuracy: {}".format(run(agent, env, num_episodes=10)))