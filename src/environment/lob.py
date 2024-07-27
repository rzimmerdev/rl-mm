import uuid
from typing import List, Dict

import matplotlib.pyplot as plt
import dxlib as dx
import numpy as np

from rbtree import RedBlackTree


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
    def __init__(self):
        self.asks = RedBlackTree()
        self.bids = RedBlackTree()
        self.orders: Dict[uuid.UUID, dx.Order] = Orders()

    def insert(self, order: dx.Order, order_id=None):
        tree = self.asks if order.side == dx.Side.BUY else self.bids
        level = tree.search(Level(order.price))

        order_id = order_id or uuid.uuid4()

        if level is tree.TNULL:
            level = Level(order.price)
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

            transactions.append(dx.Transaction(ask_order.security, quantity, ask_order.price))

            if ask_order.quantity == 0:
                self.remove(ask_level.data, self.asks)
                ask_level = self.asks.min()

            if bid_order.quantity == 0:
                self.remove(bid_level.data, self.bids)
                bid_level = self.bids.max()

            if ask_level == self.asks.TNULL or bid_level == self.bids.TNULL:
                break

        return transactions

    @property
    def mid_price(self):
        ask = self.asks.min()
        bid = self.bids.max()
        # mid price exists only if there are asks and bids
        return (ask.price + bid.price) / 2 if ask != self.asks.TNULL and bid != self.bids.TNULL else None

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
        ask_graph = ax.fill_between([price for price, _ in cumulative_asks], [quantity for _, quantity in cumulative_asks], color='red', alpha=0.3)
        bid_graph = ax.fill_between([price for price, _ in cumulative_bids], [quantity for _, quantity in cumulative_bids], color='green', alpha=0.3)
        ax.legend([ask_graph, bid_graph], ['Asks', 'Bids'])

        return fig


if __name__ == "__main__":
    book = Book()
    sec = dx.Security("AAPL")
    book.insert(dx.Order.from_data(sec, 1, 10, dx.Side.SELL))
    book.insert(dx.Order.from_data(sec, 2, 20, dx.Side.BUY))
    book.insert(dx.Order.from_data(sec, 3, 30, dx.Side.BUY))
    book.insert(dx.Order.from_data(sec, 4, 40, dx.Side.BUY))
    book.insert(dx.Order.from_data(sec, 5, 50, dx.Side.SELL))
    book.insert(dx.Order.from_data(sec, 6, 60, dx.Side.BUY))
    book.insert(dx.Order.from_data(sec, 7, 70, dx.Side.SELL))


    print("Asks:")
    for level in book.asks.inorder():
        print(level.price, [book.orders[order_id].quantity for order_id in level.orders])

    print("\nBids:")
    for level in book.bids.inorder():
        print(level.price, [book.orders[order_id].quantity for order_id in level.orders])

    print("\nOrders:")
    print(book.orders)

    print("\nOrders at price $ 1.00:")
    print(book.bids.search(Level(1.00)).orders)

    print("\nTransactions:")
    print(book.match())
    # final book after match: 2, 3, 4, 5, 6, 7

    print({order.quantity for order in book.orders.values()})
    book.plot()
    plt.show()
