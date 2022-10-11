from abc import abstractmethod

from yaaf.memory import ExperienceReplayBuffer
import numpy as np


_TEST = 0
_COUNT = 1
_DESIRED_FRACTION = 2
_ACTUAL_FRACTION = 3
_PROBABILITY = 4

_POSITIVE = 0
_NEGATIVE = 1
_ZERO = 2

_TESTS = {
    _POSITIVE: lambda reward: reward > 0,
    _NEGATIVE: lambda reward: reward < 0,
    _ZERO:     lambda reward: reward == 0
}


class BalancedExperienceReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, max_size: int, sample_size: int, positive_fraction: float, negative_fraction: float):
        super().__init__(max_size, sample_size)
        self._rewards = [
            {_TEST: _POSITIVE, _COUNT: 0, _DESIRED_FRACTION: positive_fraction},
            {_TEST: _NEGATIVE, _COUNT: 0, _DESIRED_FRACTION: negative_fraction},
            {_TEST: _ZERO,     _COUNT: 0, _DESIRED_FRACTION: 1 - positive_fraction - negative_fraction}
        ]

    def push(self, item):
        if len(self) == self._max_size:
            for reward in self._rewards:
                reward[_COUNT] -= int(_TESTS[reward[_TEST]](self._reward(self._buffer[0])))
        super().push(item)
        #if not _TESTS[_ZERO](self._reward(item)):
        #    print(item)
        for reward in self._rewards:
            reward[_COUNT] += int(_TESTS[reward[_TEST]](self._reward(item)))

    def sample(self):
        buffer_size = len(self)
        sample_size = min(self._sample_size, buffer_size)

        for reward in self._rewards:
            reward[_ACTUAL_FRACTION] = reward[_DESIRED_FRACTION] if reward[_COUNT] > 0 else 0
        fractions_sum = sum([reward[_ACTUAL_FRACTION] for reward in self._rewards])
        if fractions_sum == 0:
            return np.array([])
        for reward in self._rewards:
            reward[_PROBABILITY] = reward[_ACTUAL_FRACTION] / (fractions_sum * reward[_COUNT]) if reward[_COUNT] > 0 else 0

        p_list = []
        for item in self._buffer:
            for reward in self._rewards:
                if _TESTS[reward[_TEST]](self._reward(item)):
                    p_list.append(reward[_PROBABILITY])
                    break

        indices = np.random.choice(np.arange(buffer_size), sample_size, replace=False, p=np.array(p_list))

        return list(map(lambda i: self._buffer[i], indices))

    @staticmethod
    @abstractmethod
    def _reward(item) -> int:
        pass

