from abc import ABC

import numpy as np


class StochasticProcess(ABC):
    def sample(self, x, dt):
        raise NotImplementedError

    def __call__(self, x, dt, *args, **kwargs):
        return self.sample(x, dt)

    def simulate(self, x0, dt, n):
        n = int(n)
        x = np.zeros(n)
        x[0] = x0
        for i in range(1, n):
            x[i] = x[i - 1] + self(x[i - 1], dt)
        return x


class BrownianMotion(StochasticProcess):
    def __init__(self, mu: float = 0, sigma: float = 1):
        self.mu = mu
        self.sigma = sigma

    def sample(self, x, dt):
        return self.mu * dt + self.sigma * np.random.normal(0, 1) * np.sqrt(dt)


class OrnsteinUhlenbeck(StochasticProcess):
    def __init__(self, mu=0.0, sigma=1.0, theta=1.0):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def sample(self, x, dt):
        return self.theta * (self.mu - x) * dt + self.sigma * np.random.normal(0, 1) * np.sqrt(dt)


class CIR(StochasticProcess):
    def __init__(self, a: float, b: float, sigma: float):
        self.a = a
        self.b = b
        self.sigma = sigma

    def sample(self, x, dt):
        return self.a * (self.b - x) * dt + self.sigma * np.sqrt(x) * np.random.normal(0, 1) * np.sqrt(dt)


class AsymmetricGBM(StochasticProcess):
    def __init__(self,
                 start: float,
                 mean: float = 0,
                 volatility: float = 1,
                 spread_mean: float = 0,
                 spread_volatility=1e-3):
        self.spread_process = OrnsteinUhlenbeck(spread_mean, spread_volatility, 1e-3)
        self.spread = np.exp(self.spread_process.sample(1, 1))

        self.mid = start
        self.ask = start + self.spread
        self.bid = start - self.spread

        self.mean_process = CIR(1, mean, 1e-3)  # Cox-Ingersoll-Ross process
        self.mean = mean

        self.volatility = volatility * np.sqrt(2)

    def sample(self, x, dt):  # sample should not change state
        spread = np.exp(self.spread_process.sample(self.spread, dt))
        ask = np.random.normal((1 + self.mean) * (self.mid + self.spread), self.volatility * self.mid)
        bid = np.random.normal((1 + self.mean) * (self.mid - self.spread), self.volatility * self.mid)
        change = (ask + bid) / 2 - self.mid
        return change, (spread, ask, bid)

    def simulate(self, x0, dt, n):
        n = int(n)
        x = np.zeros(n)
        x[0] = x0
        self.mid = x0

        asks = np.zeros(n - 1)
        bids = np.zeros(n - 1)


        for i in range(1, n):
            self.mean += self.mean_process.sample(self.mean, dt)
            change, (spread, ask, bid) = self(x[i - 1], dt)
            x[i] = x[i - 1] + change

            self.spread = spread
            self.ask = ask
            self.bid = bid
            self.mid = (self.ask + self.bid) / 2

            asks[i -1] = ask
            bids[i -1] = bid

        return x, (asks, bids)


class Hawkes(StochasticProcess):
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

    def cdf(self, t):
        return 1 - np.exp(-self.intensity(t))

    def add(self, t):
        self.events = np.append(self.events, t)

    def phi(self, x):
        return np.exp(-self.beta * x)

    def intensity(self, t):
        """
        Returns the intensity of the process at time t, the intensity is the rate of events for the current time.
        """
        return self.mean + self.alpha * np.sum(self.phi(t - (self.events[self.events < t])))

    def sample(self, x, dt):
        return np.random.exponential(1 / self.intensity(x))

    def simulate(self, x0, dt, n):
        n = int(n)
        x = np.zeros(n)
        x[0] = x0
        for i in range(1, n):
            t = self(x[i - 1], dt)
            x[i] = x[i - 1] + t
            self.add(x[i])
        return x


if __name__ == "__main__":
    s = 160
    dt = 1
    T = 1 * 252  # 5 * year * dt

    asset_returns = .3  # eventually use CAPM
    asset_risk = 0.23  # eventually use CAPM

    dt_returns = (1 + asset_returns) ** (1 / 252) - 1
    dt_volatility = asset_risk / np.sqrt(252)

    agbm = AsymmetricGBM(s, dt_returns, dt_volatility)
    x, (asks, bids) = agbm.simulate(s, dt, T)

    returns = np.diff(x) / x[:-1]

    mean = np.mean(returns) * 252
    volatility = np.std(returns) * np.sqrt(252)

    print(f"Estimated values for - Mean: {mean}, Volatility: {volatility}")

    # deviation from model values
    print(f"Residuals - Mean: {mean - asset_returns}, Volatility: {volatility - asset_risk}")

    import matplotlib.pyplot as plt

    plt.plot(x)
    plt.plot(asks, alpha=.5, color='green')
    plt.plot(bids, alpha=.5, color='red')

    plt.show()
