from typing import List, Tuple

from src.lib.features import E_AGENT, E_TEAMMATE, E_OPPONENT, E_BALL


ENTITY_PREFIXES = {
    E_AGENT: "",
    E_TEAMMATE: "T",
    E_OPPONENT: "O",
    E_BALL: "ball"
}


class Feature:
    def __init__(self, name: str, data_type: int, entity_type: int = E_AGENT, entity_index: int = 0):
        self._name = name
        self._data_type = data_type
        self._entity_type = entity_type
        self._entity_index = entity_index


    @property
    def name(self) -> str:
        return self._name


    @property
    def data_type(self) -> int:
        return self._data_type


    @property
    def entity_type(self) -> int:
        return self._entity_type


    @property
    def entity_index(self) -> int:
        return self._entity_index


    def __eq__(self, other):
        return isinstance(other, Feature) and other._name == self._name and \
               other._entity_type == self._entity_type and other._entity_index == self.entity_index


    def __str__(self):
        info_list = [str(info) for info in (ENTITY_PREFIXES[self._entity_type], self._entity_index, self._name) if info]
        if len(info_list) == 3:
            info_list = [info_list[0] + info_list[1], info_list[2]]
        return "_".join(info_list)


def find(features: List[Feature], feature: Feature) -> Tuple[bool, int]:
    return (True, features.index(feature)) if feature in features else (False, 0)


def findByName(features: List[Feature], feature_name: str) -> Tuple[bool, int]:
    for f in range(len(features)):
        if features[f].name == feature_name:
            return True, f
    return False, 0


def findRequiredInputFeatures(input_features: List[Feature], required_features: List[Feature], extractor_name: str) -> List[int]:
    indices = []
    for feature in required_features:
        found, index = find(input_features, feature)
        if found:
            indices.append(index)
        else:
            exit(f"[ERROR] {extractor_name}: missing required input feature {feature}")

    return indices
