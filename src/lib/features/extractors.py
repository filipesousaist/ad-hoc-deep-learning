from src.lib.features.FeatureExtractor import FeatureExtractor

from src.lib.features.filtering.BasicFE import BasicFE
from src.lib.features.filtering.NoUnumsFE import NoUnumsFE
from src.lib.features.filtering.PlasticFE import PlasticFE
from src.lib.features.filtering.PlasticWithBallFE import PlasticWithBallFE

from src.lib.features.MemoryFE import MemoryFE
from src.lib.features.MemoryZeroFE import MemoryZeroFE
from src.lib.features.DummyFE import DummyFE
from src.lib.features.PartialObservabilityZeroFE import PartialObservabilityZeroFE


def getFeatureExtractor(name: str, num_teammates: int, num_opponents: int, *args) -> FeatureExtractor:
    return {
        "basic": BasicFE,
        "no_unums": NoUnumsFE,
        "plastic": PlasticFE,
        "plastic_with_ball": PlasticWithBallFE,
        "memory": MemoryFE,
        "memory_zero": MemoryZeroFE,
        "dummy": DummyFE,
        "partial_observability_zero": PartialObservabilityZeroFE
    }[name](num_teammates, num_opponents, *args)
