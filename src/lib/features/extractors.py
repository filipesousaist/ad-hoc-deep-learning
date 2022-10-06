from typing import List

from src.lib.features.Feature import Feature
from src.lib.features.FeatureExtractor import FeatureExtractor

from src.lib.features.filtering.BasicFE import BasicFE
from src.lib.features.filtering.NoUnumsFE import NoUnumsFE
from src.lib.features.filtering.PlasticFE import PlasticFE
from src.lib.features.filtering.PlasticWithBallFE import PlasticWithBallFE

from src.lib.features.memory.MemoryRandomFE import MemoryRandomFE
from src.lib.features.memory.MemoryZeroFE import MemoryZeroFE

from src.lib.features.partial_observability.PartialObservabilityUniformFE import PartialObservabilityUniformFE
from src.lib.features.partial_observability.PartialObservabilityByEntityFE import PartialObservabilityByEntityFE

from src.lib.features.DummyFE import DummyFE
from src.lib.features.InvalidToZeroFE import InvalidToZeroFE
from src.lib.features.ValidityFeaturesFE import ValidityFeaturesFE


def getFeatureExtractor(name: str, input_features: List[Feature], *args) -> FeatureExtractor:
    return {
        "basic": BasicFE,
        "no_unums": NoUnumsFE,
        "plastic": PlasticFE,
        "plastic_with_ball": PlasticWithBallFE,
        "memory_random": MemoryRandomFE,
        "memory_zero": MemoryZeroFE,
        "partial_observability_uniform": PartialObservabilityUniformFE,
        "partial_observability_by_entity": PartialObservabilityByEntityFE,
        "dummy": DummyFE,
        "invalid_to_zero": InvalidToZeroFE,
        "validity_features": ValidityFeaturesFE
    }[name](input_features, *args)
