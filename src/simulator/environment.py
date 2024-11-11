import warnings

import numpy as np

from src.framework import Environment, StateSpace, ActionSpace, RewardSpace
from src.simulator.dynamics import Hawkes, AsymmetricGBM
from src.lob.lob import Book, Order, Side


class MarketInterface:
    def __init__(self):
        self.lob = Book()
        self.ask = None
        self.bid = None
        self.equity = 0
        self.market_time = 0

    def clear(self):
        self.lob = Book()

    def send_update(self, order: Order):
        """
        Update the quantity of an existing order, either increase or decrease or entirely remove the order.
        If no order exists, insert a new one.
        """
        if order.side == Side.BUY:
            if self.bid and self.bid in self.lob.orders:
                return self.lob.update(self.bid, order)
            else:
                return self.lob.insert(order)
        elif order.side == Side.SELL:
            if self.ask and self.ask in self.lob.orders:
                return self.lob.update(self.ask, order)
            else:
                return self.lob.insert(order)
        else:
            raise NotImplementedError("Unknown side.")


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
            return None

    def listen(self):  # Here, listen is sync since this is a mock
        return self.lob.match(), self.market_time


class MarketInterfaceMock(MarketInterface):
    def __init__(self, market_dt=1e-3):
        super().__init__()

        self.market_time = 0
        self.market_dt = market_dt

        self.ask_means = []
        self.ask_means = []
        self.events = np.array([])
        self.previous_event = 0

        self.spread = 0.01
        self.rf = 0.005  # 0.5% risk free rate per day

        # order process
        asset_risk = 0.23  # eventually use CAPM
        dt_volatility = asset_risk / np.sqrt(50)

        self.order_event_process = Hawkes(1)
        self.order_price_process = AsymmetricGBM(100, self.rf, dt_volatility)
        self.order_quantity_distribution = np.random.poisson
        self.order_direction_distribution = np.random.choice

        self.next_event = self.order_event_process(0, 1e-3, self.events)

    def sample(self, x, spread, rf):
        next_event = self.order_event_process(self.market_time, 1e-3, self.events)
        side = Side(self.order_direction_distribution([Side.BUY.value, Side.SELL.value]))
        _, (ask, bid, spread, rf) = self.order_price_process(x, self.market_dt, spread, rf)

        quantity = self.order_quantity_distribution(1)
        price = {Side.BUY: bid, Side.SELL: ask}[side]
        order = Order.from_data(price, quantity, side)

        return next_event, order, spread, rf

    def agent_side(self, transaction):
        # decide if agent is buyer or seller based on transaction.order_id: Tuple and self.ask and self.bid
        # first, test if id 0 or id 1 of order_id is == to self.ask or self.bid
        # if it is, then the agent is the buyer or seller
        # then, decide if the agent is the buyer or seller based if == self.ask or self.bid
        if transaction.order_id[0] == self.ask or transaction.order_id[1] == self.ask:
            return Side.BUY
        elif transaction.order_id[0] == self.bid or transaction.order_id[1] == self.bid:
            return Side.SELL
        else:
            return None

    def listen(self, transactions=None):
        if self.previous_event + self.next_event <= self.market_time:
            self.market_time = np.max([self.previous_event + self.next_event, self.market_time])
            self.previous_event = self.market_time

            mid_price = self.quote() or np.random.normal(100, 1)
            mid_price = mid_price if not np.isnan(mid_price) else np.random.normal(100, 1)

            next_event, order, self.spread, self.rf = self.sample(mid_price, self.spread, self.rf)

            self.lob.insert(order)
            self.next_event = next_event
            self.events = np.append(self.events, self.market_time)
            # drop events if too large
            if len(self.events) > 500:
                self.events = self.events[-500:]
        else:
            self.market_time += self.market_dt

        transactions = self.lob.match() + (transactions or [])
        if transactions:
            for idx, transaction in enumerate(transactions):
                side = self.agent_side(transaction)
                if side == Side.BUY:
                    self.equity -= transaction.price * transaction.quantity
                elif side == Side.SELL:
                    self.equity += transaction.price * transaction.quantity
                else:
                    transactions.pop(idx)

        if not self.quote():
            self.market_time = self.previous_event + self.next_event
            transactions, self.market_time = self.listen(transactions)

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
        imbalance = Indicators.order_imbalance(lob)
        ask_imbalance = (imbalance + 1) / 2

        ask_price, bid_price = lob.top

        return ask_imbalance * (ask_price or 0) + (1 - ask_imbalance) * (bid_price or 0)

    @staticmethod
    def order_imbalance(lob: Book) -> float:
        """
        The difference between the total quantity of asks and bids
        """
        try:
            return (lob.asks.size - lob.bids.size) / (lob.asks.size + lob.bids.size)
        except TypeError:
            return 1 if lob.asks.size else -1
        except ZeroDivisionError:
            return 0

    @staticmethod
    def rsi(returns: np.ndarray, window: int = 14) -> float:
        """
        Relative Strength Index
        """
        gain = np.sum(returns[returns >= 0])
        loss = np.sum(returns[returns < 0])
        if len(returns) < window:
            return 50

        avg_gain = np.abs(gain) / len(returns)
        avg_loss = np.abs(loss) / len(returns)

        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                return 100 - 100 * (1 + avg_gain / avg_loss)
        except ZeroDivisionError:
            return 100 - 100 * (1 + avg_gain)

    @staticmethod
    def volatility(returns: np.ndarray, window: int = 10) -> float:
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                vol = np.std(returns[-window:]) / np.mean(returns[-window:])
                return vol if not np.isnan(vol) else 0
        except ZeroDivisionError:
            return 0
        finally:
            warnings.filterwarnings("default", category=RuntimeWarning)


class LobStateSpace(StateSpace):
    def __init__(self, lob_levels=5):
        """
        The state is a fabricated representation of the LOB and depends on the Environment's interpretation of the LOB.
        """
        self.lob_levels = lob_levels

    def contains(self, state: np.array):
        assert state.shape == (4 + 2 * 2 * self.lob_levels,)
        assert np.all(state[4:] >= 0)

    def indicators(self, state: np.array):
        return state[:4]

    def lob_top(self, state: np.array):
        # return first level of ask and bid
        return state[4:6], state[2 * self.lob_levels + 4:2 * self.lob_levels + 6]


class LobActionSpace(ActionSpace):
    def __init__(self):
        """
        Actions are of the form (price, quantity).
        They either update the agent's existing orders, cancel them, or insert new ones.
        """
        super().__init__()

    def contains(self, action: np.array):
        assert action is None or (
                isinstance(action, np.ndarray) and
                action.shape == (4,) and
                action[1] % 1 == 0 and
                action[3] % 1 == 0 and
                np.all(action >= 0) and
                np.all(action <= np.inf)
        ), "Invalid action"


class LobEnvironment(Environment):
    """
    The environment encapsulates the MarketInterface of a LOB.
    The state is a fabricated representation of the LOB and depends on the Environment's interpretation of the LOB.
    """
    def __init__(self,
                 api: MarketInterface,
                 action_space: ActionSpace,
                 reward_space: RewardSpace,
                 market_time: float = 1000,
                 lob_levels=5):
        super().__init__(LobStateSpace(), action_space, reward_space)
        self.api = api
        self.quotes = np.empty((0, 2))
        self.agent_quotes = np.empty((0, 3))
        self.market_time = market_time
        self.lob_levels = lob_levels

    @property
    def duration(self):
        return self.market_time

    def reset(self) -> np.array:
        self.api.clear()
        self.quotes = np.empty((0, 2))
        self.agent_quotes = np.empty((0, 3))

    def step(self, action: np.array) -> (np.array, bool):
        self.action_space.contains(action)

        if action is not None:
            ask_price, ask_quantity, bid_price, bid_quantity = action
            ask = Order.from_data(ask_price, ask_quantity, Side.SELL)
            bid = Order.from_data(bid_price, bid_quantity, Side.BUY)
            self.api.send_update(ask)
            self.api.send_update(bid)

        prev_equity = self.api.equity
        transactions, market_time = self.api.listen()

        state = self.state
        try:
            self.state_space.contains(state)
        except AssertionError:
            raise ValueError("State is not in the state space")

        self.quotes = np.append(self.quotes, np.array([[market_time, self.api.quote()]]), axis=0)

        if action is not None:
            self.agent_quotes = np.append(self.agent_quotes, np.array([[market_time, action[0], action[2]]]), axis=0)

        delta_p = self.quotes[-1][0] - self.quotes[-2][0] if len(self.quotes) > 1 else 0

        eta = 0.2
        alpha = 1e-2
        midprice = self.api.quote()
        w_t = np.sum([(transaction.price - midprice) * transaction.quantity for transaction in transactions]) if transactions else 0
        w_p = prev_equity * (delta_p - np.max(eta * delta_p, 0))

        reward = 100 - (1 / alpha) * np.exp(-alpha * (w_t + w_p))

        if abs(reward) > 100:
            print("Reward is too large")

        done = market_time > self.duration

        return reward, done

    @property
    def end(self) -> bool:
        return self.api.market_time > self.duration

    @property
    def remaining(self):
        return self.duration - self.api.market_time

    @property
    def state(self):
        asks, bids = self.api.lob.state()
        asks = np.array(asks).flatten()
        asks = np.pad(asks, (0, 2 * self.lob_levels - len(asks)))
        bids = np.array(bids).flatten()
        bids = np.pad(bids, (0, 2 * self.lob_levels - len(bids)))
        lob_state = np.append(asks, bids)
        i = Indicators
        lob_indicators = [i.micro_price(self.api.lob), i.order_imbalance(self.api.lob),
                          i.volatility(self.quotes), i.rsi(self.quotes)]

        return np.append(lob_indicators, lob_state)

    def start(self):
        self.api.market_time = 0
        self.api.clear()
        self.api.listen()
