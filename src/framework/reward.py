from abc import ABC, abstractmethod

from space import Space


class RewardSpace(Space, ABC):
    @abstractmethod
    def contains(self, action) -> bool:
        pass

