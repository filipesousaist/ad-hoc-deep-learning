import numpy as np

from src.lib.features.FeatureExtractor import FeatureExtractor
from src.lib.features.BasicFE import BasicFE
from src.lib.features.MemoryFE import MemoryFE


def getFeatureExtractor(name: str, num_teammates: int, num_opponents: int) -> FeatureExtractor:
    return {
        "basic": BasicFE,
        "memory": MemoryFE
    }[name](num_teammates, num_opponents)
