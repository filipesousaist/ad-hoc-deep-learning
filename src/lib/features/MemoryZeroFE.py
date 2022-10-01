from src.lib.features.MemoryFE import MemoryFE

"""
Apply this class before any filtering FEs
"""


class MemoryZeroFE(MemoryFE):
    def reset(self):
        for i in range(self._num_features):
            self._memorized_features[i] = 0
