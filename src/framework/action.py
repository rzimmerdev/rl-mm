from abc import ABC, abstractmethod

from space import Space


class ActionSpace(Space, ABC):
    def __init__(self):
        pass

    @abstractmethod
    def contains(self, action) -> bool:
        pass
