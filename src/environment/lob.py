import uuid
from typing import List, Dict

import matplotlib.pyplot as plt
import dxlib as dx
import numpy as np

from .rbtree import RedBlackTree


class Singleton:
    def __init__(self, cls):
        self._cls = cls
        self._instance = None

    def __call__(self, *args, **kwargs):
        if self._instance is None:
            self._instance = self._cls(*args, **kwargs)
        return self._instance


@Singleton
class Orders(dict):
    def __init__(self):
        super().__init__()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def __getitem__(self, key) -> dx.Order:
        return super().__getitem__(key)


class Level:
    def __init__(self, price, orders: List[uuid.UUID] = None):
        self.price = price
        self.orders = orders or []

    def add(self, order: uuid.UUID):
        self.orders.append(order)

    def remove(self, order: uuid.UUID):
        self.orders.remove(order)

    def __eq__(self, other: 'Level'):
        return self.price == other.price

    def __lt__(self, other: 'Level'):
        return self.price < other.price

    def __gt__(self, other: 'Level'):
        return self.price > other.price

    def top(self) -> uuid.UUID:
        return self.orders[0]

    def pop(self) -> uuid.UUID:
        return self.orders.pop()


class Book:
    def __init__(self, tick=2):
        self.asks = RedBlackTree()
        self.bids = RedBlackTree()
        self.tick = tick
        self.orders: Dict[uuid.UUID, dx.Order] = Orders()
        self.prices = []

    def insert(self, order: dx.Order, order_id=None):
        tree = self.bids if order.side == dx.Side.BUY else self.asks
        price = np.round(order.price, self.tick)
        level = tree.search(Level(price))

        order_id = order_id or uuid.uuid4()

        if level is tree.TNULL:
            level = Level(price)
            tree.insert(level)

        self.orders[order_id] = order
        level.add(order_id)

    def remove(self, level: Level, tree: RedBlackTree):
        self.orders.pop(level.pop())

        if not level.orders:
            tree.remove(level)

    def match(self) -> List[dx.Transaction]:
        transactions = []
        ask_level = self.asks.min()
        bid_level = self.bids.max()

        if ask_level == self.asks.TNULL or bid_level == self.bids.TNULL:
            return transactions

        while ask_level.price <= bid_level.price:
            ask_order = self.orders[ask_level.top()]
            bid_order = self.orders[bid_level.top()]

            quantity = min(ask_order.quantity, bid_order.quantity)
            ask_order.quantity -= quantity
            bid_order.quantity -= quantity

            transactions.append(dx.Transaction(ask_order.security, quantity, np.round(ask_level.price, self.tick)))

            if ask_order.quantity == 0:
                self.remove(ask_level.data, self.asks)
                # if the level change, remove previous level from tree since it is empty
                ask_level = self.asks.min()

            if bid_order.quantity == 0:
                self.remove(bid_level.data, self.bids)
                bid_level = self.bids.max()

            if ask_level == self.asks.TNULL or bid_level == self.bids.TNULL:
                break

        if self.mid_price:
            self.prices.append(self.micro_price)

        return transactions

    @property
    def mid_price(self):
        ask = self.asks.min()
        bid = self.bids.max()
        # mid price exists only if there are asks and bids
        return (ask.price + bid.price) / 2 if ask != self.asks.TNULL and bid != self.bids.TNULL else None

    @property
    def order_imbalance(self):
        """Order imbalance"""
        return (self.asks.size - self.bids.size) / (self.asks.size + self.bids.size) \
            if self.asks.size + self.bids.size else 0

    @property
    def rsi(self):
        """Relative Strength Index"""
        returns = np.diff(self.prices) / self.prices[:-1]
        gain = np.sum(returns[returns >= 0])
        loss = np.sum(returns[returns < 0])

        if len(returns):
            avg_gain = np.abs(gain / len(returns))
            avg_loss = np.abs(loss / len(returns))

            return 100 - 100 / (1 + avg_gain / (avg_loss + 1e-16))
        else:
            return 100

    @property
    def volatility(self):
        """Volatility"""
        return np.std(self.prices)

    def current_volatility(self, window=10):
        """Current volatility"""
        return np.std(self.prices[-window:]) if len(self.prices) > window else 0

    def state(self, levels=5):
        # return 5 levels of asks and bids
        # (level_price, level_quantity) for each level < levels
        # for bids and asks
        # traverse both self.ask and self.bids trees

        # asks
        ask_levels = []
        for level in self.asks.inorder():
            quantity = sum([self.orders[order_id].quantity for order_id in level.orders])
            ask_levels.append((level.price, quantity))

        # bids
        bid_levels = []
        for level in reversed(self.bids.inorder()):
            quantity = sum([self.orders[order_id].quantity for order_id in level.orders])
            bid_levels.append((level.price, quantity))

        return ask_levels, bid_levels

    @property
    def micro_price(self):
        """Order imbalance weighted mid price"""
        ask_imbalance = self.asks.size / (self.asks.size + self.bids.size) if self.asks.size + self.bids.size else 0
        bid_imbalance = self.bids.size / (self.asks.size + self.bids.size) if self.asks.size + self.bids.size else 0

        ask_price = self.asks.min().price if self.asks.size else 0
        bid_price = self.bids.max().price if self.bids.size else 0

        return ask_imbalance * ask_price + bid_imbalance * bid_price

    def plot(self, n=None):
        cumulative_asks = []
        cumulative_bids = []

        cumulative_quantity = 0
        for current_level in self.asks.inorder()[:n if n else self.asks.size]:
            quantities = sum([self.orders[order_id].quantity for order_id in current_level.orders])
            cumulative_quantity += quantities
            cumulative_asks.append((current_level.price, cumulative_quantity))

        cumulative_quantity = 0
        for current_level in reversed(self.bids.inorder()[-(n if n else self.bids.size):]):
            quantities = sum([self.orders[order_id].quantity for order_id in current_level.orders])
            cumulative_quantity += quantities
            cumulative_bids.append((current_level.price, cumulative_quantity))

        # Reverse cumulative_bids to plot correctly
        cumulative_bids.reverse()

        # Plot chart with grouped asks and bids
        fig, ax = plt.subplots()

        # Plot grouped asks with cumulative quantities
        ax.set_xlabel('Price')
        ax.set_ylabel('Cumulative Quantity')
        ax.set_title('Order Book with Cumulative Volume')

        # Fix legend colors
        # ask_line = plt.Line2D((0, 1), (0, 0), color='red', linewidth=3)
        # bid_line = plt.Line2D((0, 1), (0, 0), color='blue', linewidth=3)
        # Use area graphs
        ask_graph = ax.fill_between([price for price, _ in cumulative_asks],
                                    [quantity for _, quantity in cumulative_asks], color='red', alpha=0.3)
        bid_graph = ax.fill_between([price for price, _ in cumulative_bids],
                                    [quantity for _, quantity in cumulative_bids], color='green', alpha=0.3)
        ax.legend([ask_graph, bid_graph], ['Asks', 'Bids'])

        return fig


if __name__ == "__main__":
    book = Book()
    sec = dx.Security("AAPL")
    book.insert(dx.Order.from_data(sec, 1, 10, dx.Side.BUY))

    for i in range(1000):
        random_price = np.random.randint(1, 99)
        random_quantity = np.random.randint(1, 5)
        random_direction = dx.Side.BUY if np.random.rand() > 0.5 else dx.Side.SELL

        book.insert(dx.Order.from_data(sec, random_price, random_quantity, random_direction))

    print("\nOrders at price $ 1.00:")
    print(book.bids.search(Level(1.00)).orders)

    print("\nTransactions:")
    print(book.match())
    print("\nFinal Book:")
    print("Asks:")
    for level in book.asks.inorder():
        print(level.price, sum([book.orders[order_id].quantity for order_id in level.orders]))

    print("\nBids:")
    for level in book.bids.inorder():
        print(level.price, sum([book.orders[order_id].quantity for order_id in level.orders]))

    book.plot()
    plt.show()
