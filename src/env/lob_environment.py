from typing import List, Dict
from uuid import UUID

from env.simulators.order_simulator import OrderSimulator
from env.lob import LimitOrderBook, Transaction, Order


class LobEnvironment:
    def __init__(
            self,
            tick_size=2,

    ):
        self.order_simulator = OrderSimulator(
            starting_value, risk_free_mean, risk_free_std, volatility, spread_mean, spread_std
        )

        self.tick_size = tick_size
        self.lob = LimitOrderBook(tick_size)
        self.historical_variables = {
            "cash": [],
            "position": [],
            "value": [],
        }
        self.orders: Dict[UUID, Order] = {}

    def reset(self):
        self.order_simulator.reset()
        self.lob = LimitOrderBook(self.tick_size)

    def get_side(self, transaction: Transaction):
        buyer_id = transaction.buyer
        seller_id = transaction.seller

        buyer_order = self.orders.get(buyer_id)
        seller_order = self.orders.get(seller_id)

        if buyer_order is None and seller_order is None:
            return None
        else:
            return "bid" if buyer_order is not None else "ask"

    def _process_transactions(self, transactions: List[Transaction]):
        cash = self.historical_variables["cash"][-1]
        position = self.historical_variables["position"][-1]
        for transaction in transactions:
            # given if agent is buying or selling, update the cash and position
            # transaction has: seller id, buyer id, price, quantity
            side = self.get_side(transaction)
            if side is None:
                continue
            cash += transaction.price * transaction.quantity * (-1 if side == "bid" else 1)
            position += transaction.quantity * (-1 if side == "bid" else 1)

        return cash, position


    def _historical_pnl(self):
        if len(self.historical_variables["cash"]) < 2:
            return 0
        last_portfolio = self.historical_variables["cash"][-2] + self.historical_variables["position"][-2] * \
                         self.historical_variables["value"][-2]
        current_portfolio = self.historical_variables["cash"][-1] + self.historical_variables["position"][-1] * \
                            self.historical_variables["value"][-1]
        return current_portfolio - last_portfolio

    def _simulate(self):


    def step(self, action):
        self.lob.send_order(action)

        transactions = []

        orders = self.order_simulator.step(self.midprice())
        for order in orders:
            self.lob.send_order(order)
            transactions += self.lob.send_order(order)

        cash, position = self._process_transactions(transactions)

        self.historical_variables["cash"].append(cash)
        self.historical_variables["position"].append(position)
        self.historical_variables["value"].append(self.midprice())
