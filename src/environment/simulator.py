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

    @property
    def price(self):
        return self.lob.mid_price if self.lob.mid_price else self.start_price

    def reset(self):
        self.lob = Book()
        self.sampler.reset()
        self.current_time = 0

    class Action:
        def __init__(self, spread, size):
            self.spread = spread
            self.size = size

    class Reward(float):
        @staticmethod
        def utility(x: float, alpha: float = 1) -> float:  # CARA utility function
            return 1 - 1 / alpha * np.exp(-alpha * x)

        def __new__(cls, realized_pnl, Q, delta_mp, nu=0.6, alpha=1):
            return cls.utility(
                realized_pnl + Q * delta_mp - np.maximum(0, nu * Q * delta_mp),
                alpha)

    def step(self, action: Action, min_quantity=10) -> Reward:
        spread = action.spread
        size = action.size
        price = self.price + spread

        order = dx.Order.from_data(
            self.security,
            price,
            size,
            (1 if np.random.rand() > 0.5 else -1)
        )

        mp = self.lob.mid_price

        if order.quantity >= min_quantity:
            self.lob.insert(order)
            transactions = self.lob.match()



        return

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
        return self.lob, self.price, self.current_time

    def run(self, total_time, dt=1):
        starting_t = self.current_time

        with tqdm(total=total_time) as pbar:
            next_event_time = self.sampler(self.current_time)
            while (self.current_time - starting_t) < total_time:
                prev_time = self.current_time

                if next_event_time < dt or np.random.rand() > self.sampler.cdf(self.current_time + dt):
                    self.current_time += next_event_time

                    spread = np.random.normal(self.spread_mean, 1)
                    size = np.random.poisson(self.size_mean)
                    price = self.price + spread

                    order = dx.Order.from_data(
                        self.security,
                        price,
                        size,
                        (1 if np.random.rand() > 0.5 else -1)
                    )

                else:
                    self.current_time += dt

                    order = dx.Order.from_data(
                        self.security,
                        self.price,
                        1,
                        (1 if np.random.rand() > 0.5 else -1)
                    )

                self.lob.insert(order)
                self.lob.match()

                self.sampler.register_event(self.current_time)
                next_event_time = self.sampler(self.current_time)
                pbar.update(self.current_time - prev_time)

        return self.sampler.events


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
