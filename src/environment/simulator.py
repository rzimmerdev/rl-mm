from abc import ABC

import numpy as np
from tqdm import tqdm
import dxlib as dx

from .lob import Book


class BaseEnvironment:
    def __init__(self):
        pass

    @property
    def state(self):
        return None


class Hawkes:
    def __init__(self, mean=.1, alpha=.2, beta=.1):
        """
        Hawkes process with exponential kernel

        Args:
            mean: base rate of the process, the larger the mean, the more events
        """
        self.mean = mean
        self.alpha = alpha
        self.beta = beta
        self.events = np.array([])

    def reset(self):
        self.events = np.array([])

    def register_event(self, t):
        self.events = np.append(self.events, t)

    def phi(self, x):
        return self.alpha * self.beta * np.exp(-self.beta * x)

    def intensity(self, t):
        """
        Returns the intensity of the process at time t, the intensity is the rate of events for the current time.
        """
        return self.mean + np.sum(self.phi(t - (self.events[self.events < t])))

    def __call__(self, t):
        """
        Sample the time until the next event
        """
        try:
            return np.random.exponential(1 / self.intensity(t))
        except ZeroDivisionError:
            return np.inf

    def cdf(self, t):
        return 1 - np.exp(-self.intensity(t))


class Agent:
    def __init__(self):
        self.Q = 0
        self.c = 0


class Action:
    def __init__(self, spread, size):
        self.spread = spread
        self.size = size


class Reward(float):
    @staticmethod
    def utility(x: float, alpha: float = 1) -> float:  # CARA utility function
        return 1 - 1 / alpha * np.exp(-alpha * x)

    def __call__(self, realized_pnl, Q, delta_mp, nu=0.6, alpha=1):
        return self.utility(realized_pnl + Q * delta_mp - np.maximum(0, nu * Q * delta_mp), alpha)


class Environment(BaseEnvironment):
    def __init__(self, sampler, spread_mean, size_mean, start_price=100):
        super().__init__()
        self.security = dx.Security("AAPL")
        self.current_time = 0
        self.start_price = start_price
        self.lob = Book()
        self.sampler = sampler
        self.spread_mean = spread_mean
        self.size_mean = size_mean
        self.agent = Agent()
        self.reward = Reward()

    @property
    def price(self):
        return self.lob.mid_price if self.lob.mid_price else self.start_price

    def reset(self):
        self.lob = Book()
        self.sampler.reset()
        self.current_time = 0

    @property
    def state(self):
        """
        state = (o, x, t)
        o := order book state
        x := agent position state
        t := time

        o = (i, r, sigma, mp, lob_n)
        i := order book imbalance
        r := rsi
        sigma := volatility
        mp := micro price
        lob_n := (p, q) for n levels for both asks and bids

        x = (q, p)
        Q := agent inventory
        c := agent cash

        t := time
        """
        o = (
            self.lob.order_imbalance,
            self.lob.rsi,
            self.lob.current_volatility(),
            self.lob.micro_price,
            *self.lob.state(levels=5)
        )

        x = (
            self.agent.Q,
            self.agent.c
        )

        return o, x, self.current_time

    def step(self, dt, next_event_time):
        if next_event_time < dt or np.random.rand() > self.sampler.cdf(self.current_time + dt):
            spread = np.random.normal(self.spread_mean, 1)
            size = np.random.poisson(self.size_mean - 1) + 1
            price = self.price + spread

            order = dx.Order.from_data(
                self.security,
                price,
                size,
                (1 if np.random.rand() > 0.5 else -1)
            )
            self.lob.insert(order)
            transactions = self.lob.match()

            self.current_time += next_event_time
            self.sampler.register_event(self.current_time)
            return self.sampler(self.current_time), transactions

        else:
            self.current_time += dt

            return next_event_time, None

    def run(self, total_time, dt=1):
        starting_t = self.current_time

        rewards = []

        with tqdm(total=total_time) as pbar:
            prev_time = self.current_time

            prev_ask = 0
            prev_bid = 0

            next_event_time, transactions = self.step(dt, self.sampler(self.current_time))

            # Tau = {s_0, a_0, r_1, s_1, a_1, ...}
            while (self.current_time - starting_t) < total_time:
                # at this point, either:
                # 1. an event happened and the state changed (since lob changed and state contains lob)
                # 2. no event happened, time changed discretely (state contains time)

                # in both cases, the agent will read state and decide whether to send an order or not if order is
                # sent, a new event happened "technically", and the next_event_time is resampled if no order is sent,
                # the previously sampled event is statistically still the next event

                # what the agent does can be:
                # 1. agent sends order and becomes next event
                #
                # 2. agent does not send order, and considering the scenarios above:
                # in scenario 1: next_event_time is already the next event, so pass
                # in scenario 2: next_event_time is still the next event,
                #                and current_time already increased by dt, so pass
                # print(self.state)

                if np.random.rand() < 0.5:  # agent sends order and becomes next event
                    # -> reset next_event_time
                    next_event_time = self.sampler(self.current_time)
                    # send order
                else:
                    pass

                prev_ask = self.lob.asks.min().price if self.lob.asks.min().data is not None else prev_ask
                prev_bid = self.lob.bids.max().price if self.lob.bids.max().data is not None else prev_bid

                # TODO: Step 2 - r_{t+1}
                reward = self.reward(
                    realized_pnl=prev_ask - prev_bid,
                    Q=self.agent.Q,
                    delta_mp=(self.lob.micro_price - self.lob.prices[-1]) if self.lob.prices else 0
                )
                print(reward)

                # Step 3 - s_{t+1}
                # -> update state
                next_event_time, transactions = self.step(dt, next_event_time)

                # TODO: Step 2 - r_{t+1}
                # use transactions (delta_ask, q_ask, delta_bid, q_bid),
                # prev_micro_price, current micro price, current agent position
                # to calculate reward and store

                pbar.update(self.current_time - prev_time)

        return rewards


if __name__ == "__main__":
    hawkes = Hawkes(.1, 0.2, .2)
    sim = Environment(hawkes, 0.5, 1)
    times = sim.run(1000, dt=1)

    # plot hawkes intensities
    import matplotlib.pyplot as plt

    space = np.arange(0, 100, 0.1)
    lda = [hawkes.intensity(t) for t in space]

    plt.plot(
        space,
        lda,
        label="Hawkes intensity",
    )
    plt.legend()
    plt.show()

    # plot lob
    sim.lob.plot()
    plt.show()
