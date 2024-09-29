import uuid
from typing import List, Dict

import matplotlib.pyplot as plt
import dxlib as dx
import numpy as np

from rbtree import RedBlackTree
from order import Order, Side, Transaction


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

    def __getitem__(self, key) -> Order:
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
        self.orders: Dict[uuid.UUID, Order] = Orders()
        self.prices = []

    def insert(self, order: Order, order_id=None) -> uuid.UUID:
        tree = self.bids if order.side == Side.BUY.value else self.asks
        price = np.round(order.price, self.tick)
        level = tree.search(Level(price))

        order_id = order_id or uuid.uuid4()

        if level is tree.TNULL:
            level = Level(price)
            tree.insert(level)

        self.orders[order_id] = order
        level.add(order_id)

        return order_id

    def remove(self, level: Level, tree: RedBlackTree):
        self.orders.pop(level.pop())

        if not level.orders:
            tree.remove(level)

    def match(self) -> List[Transaction]:
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

            transactions.append(Transaction(quantity, np.round(ask_level.price, self.tick)))

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

    def get(self, order_id):
        return self.orders.get(order_id, None)

    def update(self, order_id, new_order: Order):
        # if price changes, remove the order from the tree and insert it again
        # if quantity increases, as well, but if it decreases, just update the order
        order = self.orders[order_id]

        if order.side != new_order.side:
            raise ValueError("Order side cannot be changed")

        if order.price != new_order.price or order.quantity < new_order.quantity:
            self.remove(self.bids.search(Level(order.price)), self.bids)
            self.remove(self.asks.search(Level(order.price)), self.asks)
            return self.insert(new_order, order_id)

        elif order.quantity > new_order.quantity:
            order.quantity = new_order.quantity
            return order_id

        else:
            raise ValueError("Order did not change")

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



if __name__ == "__main__":
    book = Book()
    book.insert(Order.from_data(1, 10, Side.BUY))

    for i in range(1000):
        random_price = np.random.randint(1, 99)
        random_quantity = np.random.randint(1, 5)
        random_direction = Side.BUY if np.random.rand() > 0.5 else Side.SELL

        book.insert(Order.from_data(random_price, random_quantity, random_direction))

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
