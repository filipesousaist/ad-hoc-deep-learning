import numpy as np

from src.lib.features.FeatureExtractor import FeatureExtractor
from src.lib.features.BasicFE import BasicFE
from src.lib.features.MemoryFE import MemoryFE
from src.lib.features.NoUnumsFE import NoUnumsFE


def getFeatureExtractor(name: str, num_teammates: int, num_opponents: int) -> FeatureExtractor:
    return {
        "basic": BasicFE,
        "memory": MemoryFE,
        "no_unums": NoUnumsFE
    }[name](num_teammates, num_opponents)
