from typing import List

from src.lib.features.dummy.DummyFE import DummyFE
from src.lib.features import E_AGENT, E_TEAMMATE, E_OPPONENT, E_BALL
from src.lib.features.Feature import Feature, findByName
from src.lib.features.Entity import getAllEntities, getAllIndices


class DummyMimicFE(DummyFE):
    def __init__(self, input_features: List[Feature], target_num_teammates: int, target_num_opponents: int):
        self._new_features = []

        super().__init__(input_features, target_num_teammates, target_num_opponents)

    def _createObservationIndices(self):
        last_features_indices = self._getLastFeaturesIndices()
        observation_indices = getAllIndices(
            getAllEntities(self.input_features, [E_AGENT, E_BALL]),
            ignore_indices=last_features_indices
        )

        teammates = getAllEntities(self.input_features, [E_TEAMMATE])
        source_num_teammates = len(teammates)

        opponents = getAllEntities(self.input_features, [E_OPPONENT])
        source_num_opponents = len(opponents)

        extra_num_teammates = self._target_num_teammates - source_num_teammates
        if extra_num_teammates < 0:
            exit("[ERROR] DummyFE: target_num_teammates must be greater than or equal to source_num_teammates.")
        extra_num_opponents = self._target_num_opponents - source_num_opponents
        if extra_num_opponents < 0:
            exit("[ERROR] DummyFE: target_num_opponents must be greater than or equal to source_num_opponents.")

        self._addFeatureIndices(getAllIndices(teammates), source_num_teammates, extra_num_teammates)
        self._addFeatureIndices(getAllIndices(opponents), source_num_opponents, extra_num_opponents)

        observation_indices += last_features_indices

        return observation_indices

    def _getLastFeaturesIndices(self) -> List[int]:
        indices = []
        for feature_name in ("last_action_success_possible", "stamina"):
            found, index = findByName(self.input_features, feature_name)
            if found:
                indices.append(index)
        indices.sort()
        return indices

    def _addFeatureIndices(self, feature_indices: List[int], source_num_entities: int, extra_num_entities: int):
        new_indices = []

        features = {}
        found_feature_names = []
        completed_feature_names = []
        for i in feature_indices:
            new_indices.append(i)

            feature = self.input_features[i]
            if feature.name in features:
                features[feature.name].append(feature)
            else:
                features[feature.name] = [feature]
                found_feature_names.append(feature.name)
            if len(features[feature.name]) == source_num_entities:
                completed_feature_names.append(feature.name)
                if len(completed_feature_names) == len(found_feature_names):
                    for feature_name in completed_feature_names:
                        new_indices += [-1] * extra_num_entities
                        sample_feature = features[feature_name][0]
                        self._new_features += [
                            Feature(feature_name, sample_feature.data_type, sample_feature.entity_type, i)
                            for i in range(source_num_entities + 1, source_num_entities + extra_num_entities + 1)
                        ]

        return new_indices

    def _createOutputFeatures(self) -> List[Feature]:
        output_features = []
        n = 0
        for i in self._observation_indices:
            if i == -1:
                output_features.append(self._new_features[n])
                n += 1
            else:
                output_features.append(self.input_features[i])
        return output_features
