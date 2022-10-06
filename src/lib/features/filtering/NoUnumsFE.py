from typing import List

from src.lib.features.FeatureExtractor import FeatureExtractor


class NoUnumsFE(FeatureExtractor):
    def _createObservationIndices(self) -> List[int]:
        return [i for i in range(self.num_input_features) if self.input_features[i].name != "uniform_number"]
