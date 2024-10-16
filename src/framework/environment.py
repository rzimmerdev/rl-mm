from abc import ABC, abstractmethod

import numpy as np


class Environment(ABC):
    def __init__(self, state_space, action_space, reward_space):
        self.state_space = state_space
        self.action_space = action_space
        self.reward_space = reward_space

    @abstractmethod
    def step(self, action: np.array) -> (np.array, np.array, bool):
        """
        Take a step in the simulator.

        Args:
            action: The action to take.

        Returns:
            The next state, the reward, and whether the episode has ended.
        """
        pass

    @abstractmethod
    def reset(self) -> np.array:
        """
        Reset the simulator to the initial state.

        Returns:
            The initial state.
        """
        pass

    @property
    @abstractmethod
    def end(self) -> bool:
        """
        Whether the episode has ended.
        """
        pass


    @property
    @abstractmethod
    def state(self):
        """
        The current state of the environment.
        """
        pass

    @property
    def duration(self):
        """
        The duration of the episode.
        """
        raise NotImplementedError("No duration measure specified for this environment.")

    @property
    def remaining(self):
        """
        The remaining duration of the episode.
        """
        raise NotImplementedError("No remaining measure specified for this environment.")