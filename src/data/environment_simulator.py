import numpy as np
import pandas as pd

from src.data.rbtree import RedBlackTree

file_path = '2013-09-09.xlsx'

xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

selected_sheet_name = input("Enter the name of the sheet you want to select: ") or "VAGR3 BS Equity"
df = pd.read_excel(file_path, sheet_name=selected_sheet_name, header=2)


class ProcessEnvironment:
    """
    Action space: AskPrice, BidPrice, AskVolume, BidVolume
    State space: Spread, Imbalance, MidPrice
    """

    def __init__(self):
        super().__init__()
        self.data = df
        self.t = 0

        self.book = {
            'bid': RedBlackTree(),
            'ask': RedBlackTree()
        }

    def reset(self):
        self.t = 0
        self.book = {
            'bid': [],
            'ask': []
        }
        return self.data.iloc[self.t]

    def spread(self):
        if len(self.book['ask']) == 0 or len(self.book['bid']) == 0:
            return 0
        return self.book['ask'][0] - self.book['bid'][0]

    def imbalance(self):
        if len(self.book['ask']) == 0 or len(self.book['bid']) == 0:
            return 0
        return (self.book['ask'][0] - self.book['bid'][0]) / (self.book['ask'][0] + self.book['bid'][0])

    def create_order(self, price, quantity, side):
        if side == 'bid':
            self.book['bid'].insert((price, quantity))
        elif side == 'ask':
            self.book['ask'].insert((price, quantity))

    def hit(self, quantity=1):
        while quantity > 0:
            if self.book['ask'][0][1] > quantity:
                self.book['ask'][0] = (self.book['ask'][0][0], self.book['ask'][0][1] - quantity)
                quantity = 0
            else:
                quantity -= self.book['ask'][0][1]
                self.book['ask'].delete(0)

    def lift(self, quantity=1):
        while quantity > 0:
            if self.book['bid'][0][1] > quantity:
                self.book['bid'][0] = (self.book['bid'][0][0], self.book['bid'][0][1] - quantity)
                quantity = 0
            else:
                quantity -= self.book['bid'][0][1]
                self.book['bid'].pop(0)

    def hawkes(self, t, alpha, beta, mu, events):
        return mu + alpha * np.sum(np.exp(-beta * (t - events)))

    def wiener(self, t, mu, sigma):
        return mu + sigma * np.random.randn()

    def poisson(self, t, lam):
        return np.random.poisson(lam)

    def step(self, action):
        # Use Hawkes to step if at next step there is an event

        # Get the current time
        t = self.data.index[self.t]
        # Add the current state to the events
        # Get the current intensity
        intensity = self.hawkes(t, alpha=0.1, beta=0.1, mu=0.1, events=self.data.index[:self.t])

        # Generate a random number
        r = np.random.rand()

        # If the random number is less than the intensity, generate an event
        if r < intensity:
            # Generate a random price using Wiener process and quantity using Poisson process
            side = 'buy' if np.random.rand() < self.imbalance() else 'sell'
            quantity = self.poisson(t, lam=1)

            # Create the order
            if side == 'buy':
                self.lift(quantity)
            else:
                self.hit(quantity)

        # Update the time
        self.t += 1

        return self.data.iloc[self.t]

    def state(self):
        return {
            'spread': self.spread(),
            'imbalance': self.imbalance(),
            'midprice': (self.book['ask'][0][0] + self.book['bid'][0][0]) / 2
        }


# Create the LOB environment
env = ProcessEnvironment()

# View the first 10 states
for _ in range(10):
    env.step(None)
    state = env.state()
    print(state)
