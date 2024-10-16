import uuid

import numpy as np

from src.framework import Environment, StateSpace, ActionSpace, RewardSpace
from src.simulator.dynamics import Hawkes, AsymmetricGBM
from src.simulator.lob import Book, Order, Side


class MarketInterface:
    def __init__(self):
        self.lob = Book()
        self.ask = None
        self.bid = None
        self.cash = 0
        self.equity = 0
        self.market_time = 0

    def clear(self):
        self.lob = Book()

    def send_insert(self, order: Order) -> uuid.UUID:
        return self.lob.insert(order)

    def send_update(self, order: Order):
        """
        Update the quantity of an existing order, either increase or decrease or entirely remove the order.
        If no order exists, insert a new one.
        """
        if order.side == Side.BUY:
            if self.bid is None:
                self.bid = self.send_insert(order)
            else:
                self.lob.update(self.bid, order)
            return self.bid
        elif order.side == Side.SELL:
            if self.ask is None:
                self.ask = self.send_insert(order)
            else:
                self.lob.update(self.ask, order)
            return self.ask
        else:
            return

    def lob_head(self, levels: int):
        return self.lob.state(levels)

    def quote(self):
        # Simplify the LOB state to a quote.
        # This method is not necessary since user can just access lob and take the average
        # But for ease of use, the interface will almost always have a quote method
        ask_price, bid_price = self.lob.top
        try:
            return ((ask_price or 0) + (bid_price or 0)) / ((ask_price is not None) + (bid_price is not None))
        except ZeroDivisionError:
            raise ValueError("No orders in the book")

    def listen(self):  # Here, listen is sync since this is a mock
        return self.lob.match(), self.market_time


class MarketInterfaceMock(MarketInterface):
    class OrderSampler:
        def __init__(self, mean=1, std=1):
            self.mean = mean
            self.std = std
            self.process = AsymmetricGBM(100)

        def __call__(self, side_mean, side):

            return Order.from_data(np.random.normal(side_mean, self.std), np.random.poisson(self.mean), side)

    def __init__(self, market_dt=1e-3):
        super().__init__()

        # event_sampler
        self.event_sampler = Hawkes()
        self.previous_event = 0
        self.next_event = self.event_sampler(self.previous_event, 1e-3)
        self.market_time = 0
        self.market_dt = market_dt

        # size sampler Poisson
        self.order_sampler = None

    def listen(self):
        # Simulate the arrival of new orders
        if self.previous_event + self.next_event < self.market_time:
            self.market_time = self.previous_event + self.next_event
            self.previous_event = self.market_time

            self.next_event = self.event_sampler(self.market_time, 1e-3)
        else:
            self.market_time += self.market_dt

        transactions = self.lob.match()

        # filter out transactions to return only transactions where user is involved
        transactions = [transaction for transaction in transactions if transaction.order_id in [self.ask, self.bid]]

        return transactions, self.market_time


class Indicators:
    # Additional fabricated interpretations of the LOB states.
    # Micro price, order imbalance, and RSI
    def __init__(self):
        pass

    @staticmethod
    def micro_price(lob: Book) -> float:
        """
        The weighted by order size mid price
        """
        try:
            bid_imbalance = lob.bids.size / (lob.asks.size + lob.bids.size)
        except ZeroDivisionError:
            raise ValueError("No orders in the book")

        ask_price, bid_price = lob.top

        return (1 - bid_imbalance) * ask_price + bid_imbalance * bid_price

    @staticmethod
    def order_imbalance(lob: Book) -> float:
        """
        The difference between the total quantity of asks and bids
        """
        try:
            return (lob.asks.size - lob.bids.size) / (lob.asks.size + lob.bids.size)
        except ZeroDivisionError:
            raise ValueError("No orders in the book")

    @staticmethod
    def rsi(returns: np.ndarray, window: int = 14) -> float:
        """
        Relative Strength Index
        """
        try:
            gain = np.sum(returns[returns >= 0])
            loss = np.sum(returns[returns < 0])

            if len(returns):
                avg_gain = np.abs(gain / len(returns))
                avg_loss = np.abs(loss / len(returns))

                return 100 - 100 / (1 + avg_gain / (avg_loss + 1e-16))
            else:
                return 100  # No returns
        except ZeroDivisionError:
            return 100

    @staticmethod
    def volatility(returns: np.ndarray, window: int = 10) -> float:
        try:
            return np.std(returns[-window:])
        except ZeroDivisionError:
            return 0


class LobStateSpace(StateSpace):
    def contains(self, state: np.array) -> bool:
        if state is None or not isinstance(state, np.ndarray):
            return False
        return state.shape == (7,)


class LobActionSpace(ActionSpace):
    def __init__(self):
        """
        Actions are of the form (price, quantity).
        They either update the agent's existing orders, cancel them, or insert new ones.
        """
        super().__init__()

    def contains(self, action: np.array) -> bool:
        return action.shape == (4,)


class LobEnvironment(Environment):
    """
    The environment encapsulates the MarketInterface of a LOB.
    The state is a fabricated representation of the LOB and depends on the Environment's interpretation of the LOB.
    """
    def __init__(self,
                 api: MarketInterface,
                 action_space: ActionSpace,
                 reward_space: RewardSpace,
                 market_time: int = 1000):
        super().__init__(LobStateSpace(), action_space, reward_space)
        self.api = api
        self.quotes = []
        self.market_time = market_time

    @property
    def duration(self):
        return self.market_time

    def reset(self) -> np.array:
        self.api.clear()
        self.quotes = []

    def step(self, action: np.array) -> (np.array, np.array, bool):
        # If action is LOB acceptable, call update on LOB (interface for an agent to a real LOB)
        if not self.action_space.contains(action):
            raise ValueError("Action is not in the action space")

        ask_price, ask_quantity, bid_price, bid_quantity = action
        ask = Order.from_data(ask_price, ask_quantity, Side.SELL)
        bid = Order.from_data(bid_price, bid_quantity, Side.BUY)
        self.api.send_update(ask)
        self.api.send_update(bid)
        transactions, market_time = self.api.listen()

        # if not enough time has passed for market to heat up (no quote available, etc), skip
        if len(self.quotes) < 5:
            return None, 0, False

        self.quotes.append(self.api.quote())
        lob_state = self.api.lob_head(levels=5)
        returns = np.diff(self.quotes) / self.quotes[:-1]
        state = np.array([Indicators.micro_price(self.api.lob),
                          Indicators.order_imbalance(self.api.lob),
                          Indicators.volatility(returns),
                          Indicators.rsi(returns)]
                         + lob_state)

        if not self.state_space.contains(state):
            raise ValueError("State is not in the state space")

        delta_p = self.quotes[-1] - self.quotes[-2] if len(self.quotes) > 1 else 0

        # Reward = Utility(realized pnl + position)
        gamma = 0.9
        reward = -np.exp(-gamma
                         * np.sum([transaction.price * transaction.quantity for transaction in transactions])
                         * self.api.equity * delta_p)

        done = market_time > self.duration

        return state, reward, done

    @property
    def end(self) -> bool:
        return self.api.market_time > self.duration

    @property
    def remaining(self):
        return self.duration - self.api.market_time

    @property
    def state(self):
        return self.api.lob_head(levels=5)
