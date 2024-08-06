from abc import ABC

import numpy as np
import dxlib as dx
from matplotlib import pyplot as plt
from tqdm import tqdm

from lob import Book, Level


class Simulator:
    def __init__(self):
        pass

    @property
    def state(self):
        return None


class Step(ABC):
    pass


class Hawkes:
    def __init__(self, mean = 0):
        """
        Hawkes process with exponential kernel

        Args:
            mean: base rate of the process, the larger the mean, the more events
        """
        self.mean = mean
        self.events = np.array([])

    def intensity(self, t):
        return self.mean + np.sum(np.exp(-self.mean * (t - self.events)))

    def add_event(self, t):
        self.events = np.append(self.events, t)

    def interval(self, t):
        return np.random.exponential(1 / np.clip(self.intensity(t), 1e-9, 1e16))


class SimpleSimulator(Simulator):
    class SimpleEvent(Step):
        def __init__(self, dt_mean, spread_mean, size_mean, price_eps=2):
            self.hawkes = Hawkes(dt_mean)
            self.spread_mean = spread_mean
            self.size_mean = size_mean
            self.price_eps = price_eps

        def reset(self):
            self.hawkes.events = np.array([])

        def next(self, simulator: 'SimpleSimulator'):
            return self.hawkes.interval(simulator.t)

        def insert(self, simulator: 'SimpleSimulator'):
            # spread is gaussian
            spread = np.random.normal(self.spread_mean, 2)
            side = dx.Side.BUY if np.random.rand() < 0.5 else dx.Side.SELL

            # size is poisson
            quantity = np.random.poisson(self.size_mean)
            price = (simulator.lob.mid_price or simulator.start_price) + spread
            price = np.round(price, self.price_eps)

            order = dx.Order.from_data(
                security=simulator.security,
                price=price,
                side=side,
                quantity=quantity
            )

            simulator.lob.insert(order)

        def remove(self, simulator: 'SimpleSimulator'):
            if not simulator.lob.orders:
                return
            orders = list(simulator.lob.orders.keys())
            order_idx = np.random.choice(np.arange(len(orders)))

            order_id = orders[order_idx]
            order_level = simulator.lob.orders[order_id].price
            side = simulator.lob.orders[order_id].side
            tree = simulator.lob.asks if side == dx.Side.BUY else simulator.lob.bids

            level = tree.search(Level(order_level))
            level.remove(order_id)
            simulator.lob.orders.pop(order_id)

            if not level.orders:
                tree.remove(level)

    def __init__(self, dt_mean, spread_mean, size_mean, start_price=100):
        super().__init__()
        self.event = self.SimpleEvent(dt_mean, spread_mean, size_mean)
        self.security = dx.Security("AAPL")
        self.t = 0
        self.start_price = start_price
        self.lob = Book()

    def reset(self):
        self.lob = Book()
        self.t = 0
        self.event.reset()

    def run(self, T):
        starting_t = self.t
        with tqdm(total=T) as pbar:
            while (self.t - starting_t) < T:
                dt = self.event.next(self)
                if dt + self.t >= T:
                    break
                event_type = np.random.choice(["delete", "insert", "stay"], p=[0.1, 0.9, 0.0])
                if event_type == "delete":
                    self.event.remove(self)
                elif event_type == "insert":
                    self.event.insert(self)
                self.t += dt
                self.lob.match()
                pbar.update(dt)


if __name__ == "__main__":
    sim = SimpleSimulator(10, 0, 1)
    sim.run(100)
    sim.lob.plot()
    mid_price = sim.lob.mid_price
    plt.show()
