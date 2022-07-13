# Adapted from:
# https://github.com/adhoc-inesc-id/ATPO-Policy

import pickle
import os

import torch
from torch.nn import Linear, LSTM, GRU

from yaaf.agents.dqn import DQNAgent
from yaaf.memory import ExperienceReplayBuffer
from yaaf.policies import epsilon_greedy_policy
from yaaf.models import TorchModel


REPLAY_BUFFER_FILE_NAME = "replay_buffer.pkl"


class DRQN(TorchModel):
    def __init__(self,
                 type: str,
                 num_features: int,
                 num_actions: int,
                 num_layers: int,
                 hidden_sizes: int,
                 dropout: float,
                 learning_rate=0.01,
                 optimizer="adam",
                 cuda=False):
        super(DRQN, self).__init__(learning_rate, optimizer, l2_penalty=0.0, loss="mse", dtype=torch.float32, cuda=cuda)
        self._num_features = num_features
        
        if type == "lstm":
            self._input_layer = LSTM(num_features, hidden_sizes, num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        elif type == "gru":
            self._input_layer = GRU(num_features, hidden_sizes, num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        
        self._q_values = Linear(hidden_sizes, num_actions)
        self.check_initialization()

    @property
    def input_shape(self):
        return self._num_features

    def forward(self, X, hidden_last=None):
        hn_last_old, hidden = self._input_layer(X) if hidden_last is None else self._input_layer(X, hidden_last)
        q_values = self._q_values(hn_last_old)
        return q_values, hidden

    def _fit_batch(self, X, y):
        X = self.data_to_model(X, torch.float32)
        y = self.data_to_model(y, self._output_dtype)
        self._optimizer.zero_grad()
        y_hat, _ = self(X)
        loss = self._loss(y_hat, y)
        loss.backward()
        self._optimizer.step()
        return loss.detach().item()

    def predict(self, X, hidden_last=None):
        if self.training: self.eval()
        self._check_device()
        with torch.no_grad():
            X = self.data_to_model(X, torch.float32)
            Z, H = self(X) if hidden_last is None else self(X, self.tuple_to_model(hidden_last, torch.float32))
            Z = Z.detach().cpu()
            if isinstance(H, tuple):
                H = H[0].detach().cpu(), H[1].detach().cpu()
            else:
                H = H.detach().cpu()
            return Z, H
    
    def tuple_to_model(self, tup: tuple, dtype: torch.dtype):
        new_tup = ()
        for tensor in tup:
            new_tup += (self.data_to_model(tensor, dtype),)
        return new_tup


    def accuracy(self, X, y):
        """
        y = y.argmax(dim=-1)
        y_hat = self.classify(X)

        B, L = y.shape

        correct = 0
        total = B * L

        for i_seq in range(B):

            for i in range(L):

                y_true = y[i_seq, i]
                y_pred = y_hat[i_seq, i]

                aux = int(y_true == y_pred)
                correct += aux

        accuracy = correct / total
        return accuracy
        """
        return 0


class DRQNAgent(DQNAgent):
    def __init__(self, num_features, num_actions,
                 rnn="lstm",
                 num_layers=2, hidden_sizes=256, dropout=0.0,
                 learning_rate=0.01, optimizer="adam",
                 discount_factor=0.95, initial_exploration_rate=0.50, final_exploration_rate=0.05,
                 initial_exploration_steps=0, final_exploration_step=5000,
                 replay_buffer_size=100000, replay_batch_size=32,
                 target_network_update_frequency=1, trajectory_update_length=4, cuda=False):

        self.max_sequence_length = trajectory_update_length
        self._current_sequence = []

        self._num_features = num_features
        self._last_hidden = None
        network = DRQN(rnn, num_features, num_actions, num_layers, hidden_sizes, dropout, learning_rate, optimizer, cuda)

        super().__init__(
            network=network,
            num_actions=num_actions,
            discount_factor=discount_factor,
            initial_exploration_rate=initial_exploration_rate, final_exploration_rate=final_exploration_rate,
            initial_exploration_steps=initial_exploration_steps, final_exploration_step=final_exploration_step,
            target_network_update_frequency=target_network_update_frequency,
            replay_buffer=ExperienceReplayBuffer(replay_buffer_size, replay_batch_size))

    def policy(self, observation):
        target = False
        q_values, self._last_hidden = self.q_values(observation, target) if self._last_hidden is None else self.q_values(observation, target, self._last_hidden)
        q_values = q_values.cpu().numpy()
        return epsilon_greedy_policy(q_values, self.exploration_rate)

    def q_values(self, observation, target=False, last_hidden=None):
        B, L, F = 1, 1, self._num_features
        z = observation.reshape((B, L, F))
        network = self._target_network if target else self._network
        q_values, hidden = network.predict(z) if last_hidden is None else network.predict(z, last_hidden)
        q_values = q_values.reshape(self._num_actions) if q_values.shape != (self._num_actions,) else q_values
        return q_values, hidden

    def _reinforce(self, timestep):

        self.remember(timestep)
        total_sequences = len(self._replay_buffer)

        info = {
            "Exploration rate": self.exploration_rate,
            "Replay Buffer Samples": len(self._replay_buffer)
        }

        buffer_initialized = total_sequences > 0 and self.total_training_timesteps >= self._replay_start_size
        at_least_one_batch = total_sequences >= self._replay_buffer._sample_size
        time_to_update = total_sequences % self._network_update_frequency == 0
        should_update_network = buffer_initialized and time_to_update and at_least_one_batch

        if should_update_network:
            loss, accuracy = self.experience_replay()
            info["Loss"] = loss
            info["Accuracy"] = accuracy

        if len(self._current_sequence) == 0:
            self._last_hidden = None

        return info

    def remember(self, timestep):
        # timestep.info["hidden state"] = self._last_hidden
        self._current_sequence.append(timestep)
        if len(self._current_sequence) == self.max_sequence_length or timestep.is_terminal:
            self._replay_buffer.push(self._current_sequence)
            self._current_sequence = []

    def _preprocess(self, batch):

        L = self.max_sequence_length
        F = self._network.input_shape
        A = self._num_actions
        B = len(batch)
        gamma = self._discount_factor
        target = True
        target_q_fn = lambda z, last_hidden: self.q_values(z, target) if last_hidden is None else self.q_values(z, target, last_hidden)

        X = torch.zeros((B, L, F))
        y = torch.zeros((B, L, A))

        for i, trajectory in enumerate(batch):

            last_hidden = None

            for t, timestep in enumerate(trajectory):

                z = torch.from_numpy(timestep.observation)
                z_next = torch.from_numpy(timestep.next_observation)
                X[i, t] = z

                target_q_values, current_hidden = target_q_fn(z, last_hidden)
                next_target_q_values, _ = target_q_fn(z_next, current_hidden)
                target_q_update = timestep.reward if timestep.is_terminal else timestep.reward + gamma * next_target_q_values.max()
                target_q_values[timestep.action] = target_q_update
                y[i, t] = target_q_values
                last_hidden = current_hidden

        return X, y


def saveReplayBuffer(replay_buffer: ExperienceReplayBuffer, directory: str):
    with open(directory + "/" + REPLAY_BUFFER_FILE_NAME, "wb") as file:
        pickle.dump(replay_buffer, file, pickle.HIGHEST_PROTOCOL)


def loadReplayBuffer(directory: str) -> ExperienceReplayBuffer:
    path = directory + "/" + REPLAY_BUFFER_FILE_NAME
    if os.path.exists(path):
        with open(path, "rb") as file:
            return pickle.load(file)
    else:
        print(f"[INFO]: File '{path}' not found. Replay buffer not loaded.")
