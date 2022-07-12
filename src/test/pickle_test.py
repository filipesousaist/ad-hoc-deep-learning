import pickle

from yaaf.memory import ExperienceReplayBuffer

from src.lib.ATPO_policy import loadReplayBuffer, saveReplayBuffer


replay_buffer: ExperienceReplayBuffer = loadReplayBuffer("experiments/39/3/agent-state/after90episodes")

sequences = replay_buffer.all

for i in range(20):
    print(len(sequences[i]))
print(sequences[0][0])