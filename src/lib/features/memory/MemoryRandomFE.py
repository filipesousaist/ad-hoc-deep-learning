from random import random, randint

from src.lib.features import F_BOOL
from src.lib.features.memory.MemoryFE import MemoryFE


class MemoryRandomFE(MemoryFE):
    def reset(self):
        for i in range(self.num_input_features):
            r = randint(0, 1) if self.input_features[i].data_type == F_BOOL \
                else random()
            self._memorized_features[i] = 2 * r - 1
