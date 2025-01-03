import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from env.simulators import MarketSimulator


class MarketEnv(gym.Env):
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

        # Observation Space
        obs_low = np.array(
            [-np.inf] * (5 + 2 * self.n_levels), dtype=np.float32
        )  # 5 core states + bid/ask prices and quantities
        obs_high = np.array(
            [np.inf] * (5 + 2 * self.n_levels), dtype=np.float32
        )
        self.observation_space = Box(obs_low, obs_high, dtype=np.float32)

        # Action Space
        action_low = np.array([0, 0, 0, 0], dtype=np.float32)  # [bid_price, bid_quantity, ask_price, ask_quantity]
        action_high = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.action_space = Box(action_low, action_high, dtype=np.float32)

    def reset(self, **kwargs):
        self.simulator.reset()
        state = self._get_state()
        return state, {}

    def step(self, action):
        transactions = self.simulator.step(action)
        next_state = self._get_state()

        reward = self._calculate_reward(transactions)
        done = False  # Define your criteria for termination
        trunc = False  # Gym truncation is generally unused but available

        return next_state, reward, done, trunc, {}

    def _get_state(self):
        lob = self.simulator.lob
        best_ask_price, best_ask_quantity = 5, 4
        best_bid_price, best_bid_quantity = 4.5, 5
        state = [
            self._calculate_rsi(),
            self._calculate_volatility(),
            self.simulator.midprice(),
            self._calculate_inventory(),
            self._calculate_order_imbalance(),
        ]

        levels = self.n_levels
        for i in range(levels):
            ask_price, ask_quantity = [(5, 4), (5.1, 3), (5.2, 2), (5.3, 1), (5.4, 1)][i]
            bid_price, bid_quantity = [(4.5, 5), (4.4, 4), (4.3, 3), (4.2, 2), (4.1, 1)][i]
            state.extend([ask_price, ask_quantity, bid_price, bid_quantity])

        return np.array(state, dtype=np.float32)

    def _calculate_rsi(self):

    def _calculate_volatility(self):

    def _calculate_inventory(self):

    def _calculate_order_imbalance(self):

    def _calculate_reward(self, transactions):


if __name__ == "__main__":
    env = MarketEnv()
    state, _ = env.reset()
    action = np.array([1.0, 10.0, 1.2, 5.0])  # Example action
    next_state, reward, done, trunc, _ = env.step(action)
