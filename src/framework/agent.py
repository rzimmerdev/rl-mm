from abc import ABC, abstractmethod

import numpy as np

from action import ActionSpace
from state import StateSpace


class Agent(ABC):
    @abstractmethod
    def act(self, state: np.array) -> np.array:
        pass

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)
