from itertools import islice

import numpy as np

from env.simulators import MarketSimulator


class Space:
    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = low
        self.high = high
        self.shape = low.shape

    def sample(self):
        return np.random.uniform(self.low, self.high)

    def contains(self, x):
        return (self.low <= x).all() and (x <= self.high).all()


class MarketEnv:
    def __init__(
            self,
            n_levels=10,
            starting_value=100,
            risk_free_mean=0.02,
            risk_free_std=0.01,
            volatility=0.1,
            spread_mean=0.1,
            spread_std=0.01,
            dt=1,
            base_event_intensity=0.5,
            event_size_mean=1,
            risk_free_reversion=0.5,
            spread_reversion=1e-2,
            order_eps=5e-2,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.simulator = MarketSimulator(
            starting_value,
            risk_free_mean,
            risk_free_std,
            volatility,
            spread_mean,
            spread_std,
            dt,
            base_event_intensity,
            event_size_mean,
            risk_free_reversion,
            spread_reversion,
            order_eps,
        )

        self.quotes = []
        self.financial_returns = []
        self.window = int(5e2)
        self.eta = 0.01
        self.alpha = 0.01
        self.duration = 390

        self.observation_space = Space(
            np.array([0, 0, 0, 0, 0] + [0, 0] * 2 * self.n_levels),
            np.array([100, 10, 1e3, 1e3, 100] + [1e4, 1e4] * 2 * self.n_levels)
        )

        self.action_space = Space(
            np.array([0, 0, 0, 0]),
            np.array([1e2, 1e2, 1e2, 1e2])
        )

    @property
    def events(self):
        return self.simulator.market_variables["events"]

    def reset(self, **kwargs):
        self.simulator.reset()
        self.simulator.fill(10)
        state = self._get_state()
        return state, {}

    def returns(self, position, starting_position):
        return (position - starting_position) / starting_position

    def step(self, action):
        previous_midprice = self.simulator.market_variables["midprice"]
        previous_inventory = self._calculate_inventory()

        action[0] = self.simulator.midprice() - action[0]
        action[2] += self.simulator.midprice()

        transactions, position, transaction_pnl = self.simulator.step(action)

        self.quotes.append(self.simulator.market_variables["midprice"])
        inventory = self.simulator.user_variables["inventory"]

        delta_midprice = self.simulator.market_variables["midprice"] - previous_midprice
        delta_inventory = inventory - previous_inventory

        next_state = self._get_state()

        reward = self._calculate_reward(transaction_pnl, inventory, delta_inventory, delta_midprice)
        trunc = False

        if np.abs(reward) > 1e4:
            trunc = True

        done = self.simulator.market_timestep >= self.duration

        return next_state, reward, done, trunc, {}

    def _calculate_reward(self, transaction_pnl, inventory, delta_inventory, delta_midprice):
        w = self.simulator.virtual_pnl(transaction_pnl, delta_inventory) + inventory * delta_midprice - np.max(
            self.eta * inventory * delta_midprice, 0)
        return 1 - np.exp(-self.alpha * w / self.simulator.market_variables["midprice"])

    def _get_state(self):
        lob = self.simulator.lob

        bids = list(islice(lob.bids.ordered_traversal(reverse=True), self.n_levels))
        asks = list(islice(lob.asks.ordered_traversal(), self.n_levels))

        state = [
            self._calculate_rsi(),
            self._calculate_volatility(),
            self.simulator.midprice(),
            self._calculate_inventory(),
            self._calculate_order_imbalance(bids, asks),
        ]

        bids = np.array([
            [order.price, order.quantity] for node in bids for order in node.value.orders
        ]).flatten()
        bids = np.pad(bids, (0, 2 * self.n_levels - len(bids)), 'constant')

        asks = np.array([
            [order.price, order.quantity] for node in asks for order in node.value.orders
        ]).flatten()
        unpadded = len(asks)
        asks = np.pad(asks, (0, 2 * self.n_levels - len(asks)), 'constant')
        asks[unpadded::2] = 1e4

        state += list(bids) + list(asks)

        return np.array(state, dtype=np.float32)

    def _calculate_rsi(self):
        if len(self.quotes) < self.window:
            return 50
        returns = np.diff(self.quotes)[-self.window:]
        up, down = returns.clip(min=0), -returns.clip(max=0)

        avg_up = np.mean(up)
        avg_down = np.mean(down)

        rs = avg_up / avg_down

        return 100 - 100 / (1 + rs)

    def _calculate_volatility(self):
        if len(self.quotes) < self.window:
            return 0
        returns = np.diff(self.quotes)[-self.window:]
        return np.std(returns)

    def _calculate_inventory(self):
        return self.simulator.user_variables["inventory"]

    def _calculate_order_imbalance(self, bids, asks):
        if not bids or not asks:
            return 0
        bid_level = bids[0].value
        ask_level = asks[0].value
        num_bids = sum(order.quantity for order in bid_level.orders)
        num_asks = sum(order.quantity for order in ask_level.orders)

        return (num_bids - num_asks) / (num_bids + num_asks)


if __name__ == "__main__":
    env = MarketEnv()
    state, _ = env.reset()
    action = np.array([1.0, 10.0, 1.2, 5.0])  # Example action
    next_state, reward, done, trunc, _ = env.step(action)
