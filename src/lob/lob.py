import uuid
from typing import List, Dict

import numpy as np

from src.lob.rbtree import RedBlackTree, Node
from src.lob.order import Order, Side, Transaction


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
        self.orders: Dict[uuid.UUID, Order] = {}

    def insert(self, order: Order, order_id=None) -> uuid.UUID:
        tree = self.bids if order.side == Side.BUY else self.asks
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

            transaction = Transaction(np.round(ask_level.price, self.tick), quantity,
                                      (ask_level.top(), bid_level.top()))
            transactions.append(transaction)

            if ask_order.quantity == 0:
                self.remove(ask_level.data, self.asks)
                # if the level change, remove previous level from tree since it is empty
                ask_level = self.asks.min()

            if bid_order.quantity == 0:
                self.remove(bid_level.data, self.bids)
                bid_level = self.bids.max()

            if ask_level == self.asks.TNULL or bid_level == self.bids.TNULL:
                break

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
            return order_id

    @property
    def top(self):
        try:
            ask_price = self.asks.min().data.price
        except AttributeError:
            ask_price = None
        try:
            bid_price = self.bids.max().data.price
        except AttributeError:
            bid_price = None

        return ask_price, bid_price

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

        return ask_levels[:levels], bid_levels[:levels]


if __name__ == "__main__":
    book = Book()
    book.insert(Order.from_data(1, 10, Side.BUY))

    for i in range(1000):
        random_price = np.random.randint(1, 99)
        random_quantity = np.random.randint(1, 5)
        random_direction = Side.BUY if np.random.rand() > 0.5 else Side.SELL

        book.insert(Order.from_data(random_price, random_quantity, random_direction))

    print("\nOrders at price $ 1.00:")
    node = book.bids.search(Level(1.00))

    assert isinstance(node, Node)
    assert node.data is not None
    print(node.data.orders)

    print("\nTransactions:")
    print(book.match())
    print("\nFinal Book:")
    print("Asks:")
    for level in book.asks.inorder():
        print(level.price, sum([book.orders[order_id].quantity for order_id in level.orders]))

    print("\nBids:")
    for level in book.bids.inorder():
        print(level.price, sum([book.orders[order_id].quantity for order_id in level.orders]))

    # plotting book
    import matplotlib.pyplot as plt

    asks, bids = book.state()

    # accumulate quantities per side
    ask_prices, ask_quantities = zip(*asks)
    bid_prices, bid_quantities = zip(*bids)

    ask_quantities = np.cumsum(ask_quantities)
    bid_quantities = np.cumsum(bid_quantities)

    plt.plot(ask_prices, ask_quantities, label='Asks')
    plt.plot(bid_prices, bid_quantities, label='Bids')

    plt.legend()
    plt.show()
