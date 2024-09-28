import uuid
from typing import Dict

import numpy as np

from src.framework import Environment, StateSpace, ActionSpace, RewardSpace
from src.simulator.hawkes import Hawkes
from src.simulator.lob import Book, Level, Order, Orders, Side


class MarketInterface:
    def __init__(self):
        self.lob = Book()
        self.orders: Dict[uuid.UUID, Order] = Orders()
        self.order_levels: Dict[float, Level] = {}
        self.cash = 0
        self.equity = 0

    def clear(self):
        self.lob = Book()

    def send_insert(self, order: Order):
        self.lob.insert(order)

    def send_update(self, order_id: uuid.UUID | None, order: Order):
        """
        Update the quantity of an existing order, either increase or decrease or entirely remove the order.
        If no order exists, insert a new one.
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            order.quantity = order.quantity + order.quantity \
                if order.side == Side.BUY else order.quantity - order.quantity
        else:
            # insert new order
            self.send_insert(order)

    def send_update_on_price(self, price: float, quantity: int, side: Side):
        # Additional logic to call send_update when the order_id is unknown, but the price level is known
        order = Order.from_data(price, quantity, side)
        order_id = self.order_levels.get(price, None)
        self.send_update(order_id, order)

    def lob_head(self, levels: int):
        return self.lob.state(levels)

    def quote(self):
        # Simplify the LOB state to a quote.
        # This method is not necessary since user can just access lob and take the average
        # But for ease of use, the interface will almost always have a quote method
        return self.lob.mid_price


class MarketInterfaceSimulator(MarketInterface):
    def __init__(self):
        super().__init__()

        # event_sampler
        self.event_sampler = Hawkes()

        # size sampler Poisson
        self.size_sampler = 1

    def send_update_on_price(self, price: float, quantity: int, side: Side):
        transactions = super().send_update_on_price(price, quantity, side)

        # Simulate the arrival of new orders
        t = 0



class Indicators:
    # Additional fabricated interpretations of the LOB states.
    # Micro price, order imbalance, and RSI
    def __init__(self):
        pass

    @staticmethod
    def micro_price(lob: Book) -> float:
        """
        The mid price of the LOB
        """
        pass

    @staticmethod
    def order_imbalance(lob: Book) -> float:
        """
        The difference between the total quantity of asks and bids
        """
        pass

    @staticmethod
    def rsi(lob: Book, window: int = 14) -> float:
        """
        Relative Strength Index
        """
        pass


class LobStateSpace(StateSpace):
    def contains(self, state: np.array) -> bool:
        return state.shape == (1, 1)


class LobActionSpace(ActionSpace):
    def __init__(self):
        """
        Actions are of the form (price, quantity).
        They either update the agent's existing orders, cancel them, or insert new ones.
        """
        super().__init__()

    def contains(self, action: np.array) -> bool:
        return action.shape == (1, 1)


class LobEnvironment(Environment):
    """
    The environment encapsulates the MarketInterface of a LOB.
    The state is a fabricated representation of the LOB and depends on the Environment's interpretation of the LOB.
    """

    def __init__(self, api: MarketInterface, action_space, reward_space, market_close: int = 1000):
        super().__init__(LobStateSpace(), action_space, reward_space)
        self.api = api
        self.quotes = []
        self.t = 0
        self.market_close = market_close

    def reset(self) -> np.array:
        self.api.clear()
        self.quotes = []
        self.t = 0

    def step(self, action: np.array) -> (np.array, np.array, bool):
        # MDP process: Insert action into limit order book, update state. return reward

        # If action is LOB acceptable, call update on LOB (interface for an agent to a real LOB)
        assert self.action_space.contains(action)

        price, quantity, side = action
        transactions = self.api.send_update_on_price(price, quantity, side)

        # state = self.state_space.sample()  <-- This would mean the LOB's interpretation is used
        # and the environments interpretation of the LOB is the same
        lob_state = self.api.lob_head(levels=5)
        state = np.array([
                             Indicators.rsi(self.api.lob),
                             Indicators.order_imbalance(self.api.lob),
                             np.std(self.quotes[-10:]),
                             Indicators.micro_price(self.api.lob)]
                         + lob_state
                         )

        self.quotes.append(self.api.quote())

        delta_p = self.quotes[-1] - self.quotes[-2] if len(self.quotes) > 1 else 0

        # Reward = Utility(realized pnl + position)
        gamma = 0.9
        reward = -np.exp(-gamma
                         * np.sum([transaction.price * transaction.quantity for transaction in transactions])
                         * self.api.equity * delta_p)

        self.t += 1
        done = self.t > self.market_close

        return state, reward, done
