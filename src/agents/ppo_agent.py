from typing import Union

import numpy as np

from src.framework import Agent
from src.simulator.environment import LobStateSpace


class PPOAgent(Agent):
    def __init__(self):
        self.ask_price = None
        self.ask_quantity = None
        self.bid_price = None
        self.bid_quantity = None

    def act(self, state: Union[np.array]) -> np.array:
        if state is None:
            return None
        ask, bid = LobStateSpace().lob_top(state)
        if not (ask[0] < np.finfo(float).max and bid[0] > np.finfo(float).eps) or ask[0] < bid[0]:
            return None
        # ask[0] = price, ask[1] = quantity, same for bid
        ask_price = ask[0]
        bid_price = bid[0]
        if not self.ask_price or np.abs(ask_price - self.ask_price) > 1/100 * ask_price:
            self.ask_price = np.minimum(ask_price, np.finfo(float).max)
            self.ask_quantity = 1
        if not self.bid_price or np.abs(bid_price - self.bid_price) > 1/100 * bid_price:
            self.bid_price = np.maximum(bid_price, 0)
            self.bid_quantity = 1

        return np.array([self.ask_price, self.ask_quantity, self.bid_price, self.bid_quantity])

    def update(self, next_state, reward, done):
        pass