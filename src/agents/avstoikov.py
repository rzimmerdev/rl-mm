from typing import Union

import numpy as np

from src.framework import Agent
from src.simulator.environment import LobStateSpace






class AvellanedaStoikov(Agent):
    def __init__(self):
        self.ask_price = None
        self.ask_quantity = None
        self.bid_price = None
        self.bid_quantity = None

        self.risk_aversion = 0.2
        self.volatility = None
        self.liquidity = None
        self.previous_state = None
        self.quotes = []
        self.volumes = []

    @staticmethod
    def mp(state):
        ask, bid = LobStateSpace().lob_top(state)
        return (ask[0] + bid[0]) / 2

    @staticmethod
    def tob(state):
        ask, bid = LobStateSpace().lob_top(state)
        return (ask[1] + bid[1]) / 2

    @staticmethod
    def empty_state(state):
        ask, bid = LobStateSpace().lob_top(state)
        if not (ask[0] < np.finfo(float).max and bid[0] > np.finfo(float).eps) or ask[0] < bid[0]:
            return None, None
        return ask, bid

    def act(self, state: Union[np.array]) -> np.array:
        if state is None:
            return None
        ask, bid = self.empty_state(state)
        if ask is None:
            return None
        if self.previous_state is None or not self.empty_state(self.previous_state):
            self.previous_state = state
            return None
        mid_price = self.mp(state)

        # append only non-repeated mid prices
        if len(self.quotes) == 0 or mid_price != self.quotes[-1]:
            self.quotes.append(mid_price)
        self.volumes.append(np.abs(self.tob(state) - self.tob(self.previous_state)))

        if len(self.quotes) < 200:
            return None

        volatility = np.std(self.quotes[-100:])
        liquidity = np.mean(self.volumes[-100:])

        if liquidity == 0:
            return None

        spread = np.round((self.risk_aversion * volatility ** 2 / liquidity), 2)

        if spread == 0:
            return None

        ask_price = mid_price + spread
        bid_price = mid_price - spread

        # if no prices are set, or new spread is 1% different from previous spread
        if not self.ask_price or not self.bid_price or np.abs(spread - (self.ask_price - self.bid_price)) > 0.01 * spread:
            self.ask_price = np.minimum(ask_price, np.finfo(float).max)
            self.ask_quantity = 1
            self.bid_price = np.maximum(bid_price, 0)
            self.bid_quantity = 1

        self.previous_state = state

        return np.array([self.ask_price, self.ask_quantity, self.bid_price, self.bid_quantity])

    def update(self, next_state, reward, done):
        pass