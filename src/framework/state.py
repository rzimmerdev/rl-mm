from abc import ABC, abstractmethod

from space import Space


class StateSpace(Space, ABC):
    @abstractmethod
    def contains(self, state) -> bool:
        pass
