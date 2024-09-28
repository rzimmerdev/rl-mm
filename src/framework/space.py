from abc import ABC, abstractmethod


class Space(ABC):
    @abstractmethod
    def contains(self, x) -> bool:
        pass
