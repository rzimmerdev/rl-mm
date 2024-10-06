from abc import ABC, abstractmethod

import numpy as np

from .action import ActionSpace
from .state import StateSpace


class Agent(ABC):
    @abstractmethod
    def act(self, state: np.array) -> np.array:
        pass

    @abstractmethod
    def update(self, next_state: np.array, reward: np.array, done: bool):
        pass

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)
