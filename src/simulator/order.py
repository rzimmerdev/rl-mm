from enum import Enum


class Order:
    def __init__(self, price, quantity, side):
        self.price = price
        self.quantity = quantity
        self.side = side

    @classmethod
    def from_data(cls, price, quantity, side):
        return Order(price, quantity, side)


class Transaction:
    def __init__(self, price, quantity):
        self.price = price
        self.quantity = quantity

    @classmethod
    def from_data(cls, price, quantity):
        return Transaction(price, quantity)


class Side(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0
