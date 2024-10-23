from abc import ABC

import numpy as np


class StochasticProcess(ABC):
    def sample(self, x, dt, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, x, dt, *args, **kwargs):
        return self.sample(x, dt, *args, **kwargs)

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

    def sample(self, x, dt, *args, **kwargs):
        return self.mu * dt + self.sigma * np.random.normal(0, 1) * np.sqrt(dt)


class OrnsteinUhlenbeck(StochasticProcess):
    def __init__(self, mu=0.0, sigma=1.0, theta=1.0):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def sample(self, x, dt, *args, **kwargs):
        return self.theta * (self.mu - x) * dt + self.sigma * np.random.normal(0, 1) * np.sqrt(dt)


class CIR(StochasticProcess):
    def __init__(self, a: float, b: float, sigma: float):
        self.a = a
        self.b = b
        self.sigma = sigma

    def sample(self, x, dt, *args, **kwargs):
        x = max(x, 0.0)

        drift = self.a * (self.b - x) * dt
        vol = self.sigma * np.sqrt(x) * np.random.normal(0, 1) * np.sqrt(dt)

        x_new = x + drift + vol

        return max(x_new, 0.0)

class AsymmetricGBM(StochasticProcess):
    def __init__(self,
                 start: float,
                 rf_mean: float = 0,
                 volatility: float = 0.01,
                 spread_mean: float = 0.1,
                 spread_volatility=1e-3):
        self.spread_process = OrnsteinUhlenbeck(spread_mean, spread_volatility, 1e-3)
        self.rf_process = CIR(1, rf_mean, 1e-3)  # Cox-Ingersoll-Ross process

        self.ask = start + spread_mean
        self.bid = start - spread_mean

        self.volatility = volatility * np.sqrt(2)

    def sample(self, x, dt, *args, **kwargs):
        spread, rf = args
        spread = np.exp(self.spread_process.sample(spread, dt))
        rf = self.rf_process.sample(rf, dt)
        ask = np.random.normal((1 + rf) * (x + spread), self.volatility * x)
        bid = np.random.normal((1 + rf) * (x - spread), self.volatility * x)
        change = (ask + bid) / 2 - x
        return np.max([change, -x]), (ask, bid, spread, rf)

    def simulate(self, x0, dt, n):
        n = int(n)
        x = np.zeros(n)
        x[0] = x0

        asks = np.zeros(n - 1)
        bids = np.zeros(n - 1)

        rf = self.rf_process.b
        spread = self.spread_process.mu

        for i in range(1, n):
            change, (ask, bid, spread, rf) = self.sample(x[i - 1], dt, rf, spread)
            x[i] = x[i - 1] + change
            asks[i - 1] = ask
            bids[i - 1] = bid

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

    def cdf(self, t, events):
        return 1 - np.exp(-self.intensity(t, events))

    def phi(self, x):
        return np.exp(-self.beta * x)

    def intensity(self, t, events):
        """
        Returns the intensity of the process at time t, the intensity is the rate of events for the current time.
        """
        return self.mean + self.alpha * np.sum(self.phi(t - (events[events < t])))

    def sample(self, x, dt, *args, **kwargs):
        events, = args
        return np.random.exponential(1 / self.intensity(x, events))

    def simulate(self, x0, dt, n):
        n = int(n)
        x = np.zeros(n)
        x[0] = x0
        events = np.zeros(n)
        for i in range(1, n):
            t = self.sample(x[i - 1], dt, events)
            x[i] = x[i - 1] + t
            events[i] = x[i]
        return x


def main():
    s = 160
    dt = 1
    timesteps = 1 * 252  # 5 * year * dt

    asset_returns = .3  # eventually use CAPM
    asset_risk = 0.23  # eventually use CAPM

    dt_returns = (1 + asset_returns) ** (1 / 252) - 1
    dt_volatility = asset_risk / np.sqrt(252)

    agbm = AsymmetricGBM(s, dt_returns, dt_volatility)
    x, (asks, bids) = agbm.simulate(s, dt, timesteps)

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


if __name__ == "__main__":
    main()
