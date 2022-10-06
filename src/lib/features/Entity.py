from typing import List

from src.lib.features.Feature import Feature


class Entity:
    def __init__(self, typ: int, index: int):
        self._type = typ
        self._index = index
        self._feature_indices = []
        self._list_index = -1


    @property
    def type(self):
        return self._type


    @property
    def index(self):
        return self._index


    @property
    def feature_indices(self):
        return self._feature_indices


    @property
    def list_index(self):
        return self._list_index


    def setListIndex(self, list_index):
        self._list_index = list_index


    def addFeatureIndex(self, index: int):
        self._feature_indices.append(index)


    def __eq__(self, other):
        return isinstance(other, Entity) and self._type == other._type and self._index == other._index


def getAllEntities(features: List[Feature], types: List[int]) -> List[Entity]:
    entities = []
    for f in range(len(features)):
        if features[f].entity_type in types:
            entity = Entity(features[f].entity_type, features[f].entity_index)
            if entity not in entities:
                entity.setListIndex(len(entities))
                entities.append(entity)
            entities[entities.index(entity)].addFeatureIndex(f)
    return entities


def getAllIndices(entities: List[Entity], ignore_indices: List[int] = ()):
    indices = []
    for entity in entities:
        indices.extend([
            index for index in entity.feature_indices if index not in ignore_indices
        ])
    indices.sort()
    return indices
