from abc import ABC, abstractmethod
from typing import Tuple

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
# set Date as index
df.set_index('Date', inplace=True)


class Environment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action: any) -> any:
        pass

    @property
    @abstractmethod
    def state(self) -> any:
        pass


class LOBEEnvironment(Environment):
    """
    Action space: AskPrice, BidPrice, AskVolume, BidVolume
    State space: Spread, Imbalance, MidPrice
    """
    def __init__(self, frequency='1s'):
        super().__init__()
        self.data = df

        # t is the time step in the dataframe, starts at the first row
        self.t: pd.Timestamp = self.data.index[0]
        self.frequency = frequency

        self.book = {
            'bid': [],
            'ask': []
        }

        self.agent_pos = {}

    @property
    def dt(self):
        # Increment the time step
        return pd.Timedelta(self.frequency)

    def at_end(self):
        return self.t >= self.data.index[-1]

    def reset(self):
        self.t = self.data.index[0]
        self.book = {
            'bid': [],
            'ask': []
        }

    def process(self, event: pd.Series) -> Tuple[float, float] | None:
        # Date, Type, Price, Size
        # Type in ["TRADE", "BEST_BID", "BEST_ASK"]
        side = None
        if event['Type'] == 'TRADE':
            return event['Price'], event['Size']
        elif event['Type'] == 'BEST_BID':
            side = 'bid'
        elif event['Type'] == 'BEST_ASK':
            side = 'ask'
        self.create_order(event['Price'], event['Size'], side)

    def create_order(self, price: float, quantity: float, side: str):
        if side == 'bid':
            self.book['bid'].append((price, quantity))
            self.book['bid'] = sorted(self.book['bid'], key=lambda x: x[0], reverse=True)
        elif side == 'ask':
            self.book['ask'].append((price, quantity))
            self.book['ask'] = sorted(self.book['ask'], key=lambda x: x[0])

    def step(self, action: Tuple[float, float, float, float] | None) -> Tuple[float, bool]:
        self.t += self.dt
        reward = 0
        done = False

        # Read all events at time t events = self.data.loc[self.t] if self.t in self.data.index else []
        events = self.data.loc[self.t] if self.t in self.data.index else pd.DataFrame()

        for event in events.iterrows():
            index, event = event
            trade = self.process(event)

            if trade:
                price, quantity = trade
                # If trade with agent, update agent position
                # if trade price <= best bid, agent bought
                # if no book, ignore
                if len(self.book['bid']) == 0 or len(self.book['ask']) == 0:
                    continue
                elif price <= self.book['bid'][0][0]:
                    self.agent_pos['bid'] = price
                # if trade price >= best ask, agent sold
                elif price >= self.book['ask'][0][0]:
                    self.agent_pos['ask'] = price

        if self.at_end():
            done = True

        return reward, done

    @property
    def order_imbalance(self):
        if len(self.book['ask']) == 0 or len(self.book['bid']) == 0:
            return 0
        return (self.book['ask'][0][0] - self.book['bid'][0][0]) / (self.book['ask'][0][0] + self.book['bid'][0][0])

    @property
    def spread(self):
        if len(self.book['ask']) == 0 or len(self.book['bid']) == 0:
            return 0
        return self.book['ask'][0][0] - self.book['bid'][0][0]

    @property
    def mid_price(self):
        if len(self.book['ask']) == 0 or len(self.book['bid']) == 0:
            return 0
        return (self.book['ask'][0][0] + self.book['bid'][0][0]) / 2

    @property
    def state(self) -> Tuple[float, float, float]:
        return self.spread, self.order_imbalance, self.mid_price


env = LOBEEnvironment()
env.reset()

for i in range(10):
    env.step(None)
    print(env.state)
