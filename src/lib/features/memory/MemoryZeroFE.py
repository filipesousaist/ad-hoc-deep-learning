from src.lib.features.memory.MemoryFE import MemoryFE


class MemoryZeroFE(MemoryFE):
    def reset(self):
        for i in range(self.num_input_features):
            self._memorized_features[i] = 0
