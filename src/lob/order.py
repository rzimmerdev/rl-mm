from enum import Enum


class Order:
    def __init__(self, price, quantity, side):
        self.quantity = quantity
        self.price = price
        self.side = side

    @classmethod
    def from_data(cls, price, quantity, side):
        return Order(price, quantity, side)


class Transaction:
    def __init__(self, price, quantity, order_id):
        self.price = price
        self.quantity = quantity
        self.order_id = order_id

    @classmethod
    def from_data(cls, price, quantity, order_id):
        return Transaction(price, quantity, order_id)


class Side(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0
