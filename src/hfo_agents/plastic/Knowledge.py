import numpy as np
from yaaf.agents.dqn import DQNAgent

from src.lib.models.NearestObservationModel import NearestObservationModel


class Knowledge:
    def __init__(self, agent: DQNAgent, model: NearestObservationModel):
        self._agent = agent
        self._model = model


    @property
    def agent(self) -> DQNAgent:
        return self._agent


    def policy(self, features: np.ndarray) -> np.ndarray:
        return self._agent.policy(features)


    def getLoss(self, features: np.ndarray, next_features: np.ndarray) -> float:
        predicted_features = self._model.getNextObservation(features)
        return np.linalg.norm(predicted_features - next_features) / self._model.max_distance


