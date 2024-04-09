import abc
from typing import Tuple

import numpy as np
import pandas as pd

# Replace 'your_file.xlsx' with the path to your Excel file
file_path = '2013-09-09.xlsx'

# Read the Excel file and get the list of sheet names
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

# Print the sheet names
print("Available sheets:")
for sheet_name in sheet_names:
    print(sheet_name)

# Select a sheet
selected_sheet_name = input("Enter the name of the sheet you want to select: ") or "VAGR3 BS Equity"

# Read the selected sheet
df = pd.read_excel(file_path, sheet_name=selected_sheet_name, header=2)

# Print the first 5 rows of the selected sheet
print(df.head())
print(df.columns)
print(df.dtypes)
print(df.shape)
print(df.describe())
print(df.index[0])


class Environment(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def step(self, action: any) -> any:
        pass

    @property
    @abc.abstractmethod
    def state(self) -> any:
        pass

# LOB environment for RL agent
class LOBEnvironment(Environment):
    """
    Action space: AskPrice, BidPrice, AskVolume, BidVolume
    State space: Spread, Imbalance, MidPrice
    """
    def __init__(self):
        super().__init__()
        self.data = df
        self.t = 0

        self.book = {
            'bid': [],
            'ask': []
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
            self.book['bid'].append((price, quantity))
        elif side == 'ask':
            self.book['ask'].append((price, quantity))

    def hit(self, quantity=1):
        while quantity > 0:
            if self.book['ask'][0][1] > quantity:
                self.book['ask'][0] = (self.book['ask'][0][0], self.book['ask'][0][1] - quantity)
                quantity = 0
            else:
                quantity -= self.book['ask'][0][1]
                self.book['ask'].pop(0)

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
env = LOBEnvironment()

# View the first 10 states
for _ in range(10):
    env.step(None)
    state = env.state()
    print(state)
