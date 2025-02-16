import numpy as np
import scipy


class AvStoikov:
    def __init__(self, n_levels=5):
        self.quotes = []
        self.n_levels = n_levels

    def get_action(self, spread):
        # avellaneda stoikov closed form equation

        # if quotes > 15, use 15 most recent quotes for volatility, else, use all quotes
        quotes = self.quotes[-15:] if len(self.quotes) > 15 else self.quotes
        volatility = np.std(quotes)

        return volatility / np.sqrt(2) * self.erf(1 / 2 * (1 + spread / volatility))

    def erf(self, x):
        # \text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt
        # inverse error function
        return scipy.special.erf(x)


class TestStoikov():
    def __init__(self):
        self.agent = AvStoikov()
        self.inventory = 0
        self.cash = 0

    def observe(self, midprice, spread):
        delta = (midprice - self.agent.quotes[-1]) if self.agent.quotes else 0
        self.agent.quotes.append(midprice)

        # ignore market impact and use midprice
        if len(self.agent.quotes) < 5:
            return
        action = self.agent.get_action(spread)

        # if either best bid or best ask match the agent action, update its pnl
        if delta > action:
            # got filled on ask side
            self.inventory -= 1
            self.cash += midprice - delta + action
        elif delta < -action:
            # got filled on bid side
            self.inventory += 1
            self.cash -= midprice - delta - action

    def pnl(self):
        return self.cash + ((self.inventory * self.agent.quotes[-1]) if self.agent.quotes else 0)

    def reset(self):
        self.agent = AvStoikov()
        self.inventory = 0
        self.cash = 0
