from typing import List

import numpy as np

from env.lob import LimitOrderBook, Order, Transaction
from env.dynamics import GeometricBrownianMotion, CoxIngersollRoss, OrnsteinUhlenbeck, Hawkes


class MarketSimulator:
    def __init__(
            self,
            starting_value,
            risk_free_mean=0.02,
            risk_free_std=0.01,
            volatility=0.1,
            spread_mean=0.1,
            spread_std=0.01,
            dt=1,
            base_event_intensity=0.5,
            event_size_mean=1,
            risk_free_reversion: float = 0.5,
            spread_reversion: float = 1e-2,
            order_eps: float = 2e-2,
            starting_cash: float = 0,
            starting_inventory: float = 0
    ):
        self.dt = dt
        self.risk_free_mean = risk_free_mean
        self.risk_free_std = risk_free_std
        self.spread_mean = spread_mean
        self.spread_std = spread_std

        bid_center = self.risk_free_mean - self.spread_mean / 2 / self.dt
        ask_center = self.risk_free_mean + self.spread_mean / 2 / self.dt

        self.bid_process = GeometricBrownianMotion(bid_center, volatility)
        self.ask_process = GeometricBrownianMotion(ask_center, volatility)

        self.risk_free_process = CoxIngersollRoss(
            risk_free_reversion, risk_free_mean, risk_free_std)
        self.spread_process = OrnsteinUhlenbeck(
            spread_reversion, spread_mean, spread_std)

        self.event_process = Hawkes(
            base_event_intensity, branching_ratio=1e-1, decay=0.3)

        self.quantity_distribution = np.random.poisson
        self.event_size_distribution = np.random.poisson
        self.event_size_mean = event_size_mean

        self.tick_size = 2
        self.lob: LimitOrderBook = LimitOrderBook(self.tick_size)

        self.starting_value = starting_value
        self.risk_free = risk_free_mean

        self.market_variables = {
            "last_transaction": self.starting_value,
            "midprice": [self.starting_value],
            "risk_free_rate": self.risk_free,
            "spread": self.spread_mean,
            "timestep": 0,
            "events": []
        }

        self.order_eps = order_eps
        self.user_variables = {
            "bid_price": None,
            "ask_price": None,
            "bid_quantity": None,
            "ask_quantity": None,
            "ask_order": None,
            "bid_order": None,
            "cash": starting_cash,
            "inventory": starting_inventory
        }

        self.next_event = self.event_process.sample(
            0, self.dt, np.array(self.market_variables['events']))
        self.previous_event = 0

    def reset(self):
        self.bid_process.mean = self.risk_free_mean - self.spread_mean / 2 / self.dt
        self.ask_process.mean = self.risk_free_mean + self.spread_mean / 2 / self.dt
        self.lob = LimitOrderBook(self.tick_size)
        self.market_variables = {
            "last_transaction": self.starting_value,
            "midprice": [self.starting_value],
            "risk_free_rate": self.risk_free,
            "spread": self.spread_mean,
            "timestep": 0,
            "events": []
        }
        self.user_variables = {
            "bid_price": None,
            "ask_price": None,
            "bid_quantity": None,
            "ask_quantity": None,
            "ask_order": None,
            "bid_order": None,
            "cash": 0,
            "inventory": 0
        }
        self.next_event = self.event_process.sample(0, self.dt, np.array([]))

    def fill(self, n):
        # generate n events to fill the order book
        for _ in range(n):
            orders = self._sample_orders(self.midprice())
            for order in orders:
                self.lob.send_order(order)
            self.market_variables['midprice'].append(self.midprice())
            self.market_variables['timestep'] += self.dt

    def midprice(self):
        best_ask = self.lob.asks.bottom(
        ).value.price if self.lob.asks.bottom() is not None else None
        best_bid = self.lob.bids.top().value.price if self.lob.bids.top() is not None else None

        if best_ask is not None and best_bid is not None:
            return (best_ask + best_bid) / 2
        # else, return mean of last 5 midprices
        return np.mean(self.market_variables['midprice'][-5:])

    def _sample_orders(self, price):
        bid_size = self.event_size_distribution(self.event_size_mean)
        ask_size = self.event_size_distribution(self.event_size_mean)

        bids = self.bid_process.sample(price, self.dt, bid_size)
        asks = self.ask_process.sample(price, self.dt, ask_size)

        bids_quantity = np.array(self.quantity_distribution(size=bid_size) + 1)
        asks_quantity = np.array(self.quantity_distribution(size=ask_size) + 1)

        orders = [Order(price, quantity, 'ask') for price, quantity in zip(asks, asks_quantity)] + \
                 [Order(price, quantity, 'bid')
                  for price, quantity in zip(bids, bids_quantity)]

        return np.array(orders)

    def set_order(self, bid=None, ask=None):
        # replace price and quantity of desired order.
        # if price of existing market order is order_eps different from the desired order, then cancel the existing order and replace
        if bid:
            self.user_variables["bid_price"] = bid.price
            self.user_variables["bid_quantity"] = bid.quantity

            existing_order = self.user_variables.get("bid_order", None)
            if existing_order:
                existing_order = self.lob.orders.get(existing_order.uuid, None)
            if existing_order is None:
                self.lob.send_order(bid)
                self.user_variables["bid_order"] = bid
            elif abs(existing_order.price - bid.price) > self.order_eps:
                self.lob.cancel_order(existing_order.uuid)
                self.lob.send_order(bid)
                self.user_variables["bid_order"] = bid

        if ask:
            self.user_variables["ask_price"] = ask.price
            self.user_variables["ask_quantity"] = ask.quantity

            existing_order = self.user_variables.get("ask_order", None)
            if existing_order:
                existing_order = self.lob.orders.get(existing_order.uuid, None)
            if existing_order is None:
                self.lob.send_order(ask)
                self.user_variables["ask_order"] = ask
            elif abs(existing_order.price - ask.price) > self.order_eps:
                self.lob.cancel_order(existing_order.uuid)
                self.lob.send_order(ask)
                self.user_variables["ask_order"] = ask

    def _participated_side(self, transaction):
        if self.user_variables["bid_order"] and transaction.buyer == self.user_variables["bid_order"].uuid:
            return 'bid'
        elif self.user_variables["ask_order"] and transaction.seller == self.user_variables["ask_order"].uuid:
            return 'ask'
        return None

    def _participated_transactions(self, transactions) -> tuple[list[Transaction], list[Transaction]]:
        bids: List[Transaction] = []
        asks: List[Transaction] = []
        for transaction in transactions:
            side = self._participated_side(transaction)
            if side == 'bid':
                bids.append(transaction)
            elif side == 'ask':
                asks.append(transaction)
        return bids, asks

    def _calculate_pnl(self, transactions):
        bids, asks = self._participated_transactions(transactions)
        if not bids and not asks:
            return 0, 0
        delta_inventory = sum([bid.quantity for bid in bids]) - \
            sum([ask.quantity for ask in asks])
        transaction_pnl = -sum([bid.price * bid.quantity for bid in bids]) + sum(
            [ask.price * ask.quantity for ask in asks])
        return transaction_pnl, delta_inventory

    def best(self, side):
        best = self.lob.bids.top() if side == 'bid' else self.lob.asks.bottom()
        return best.value.price if best is not None else self.midprice()

    def virtual_pnl(self, pnl, quantity):
        # pnl + (inventory value - liquidation premium) -> liquidation premium assumes no large lob walking
        best_price = self.best('bid' if quantity > 0 else 'ask')
        liquidation_pnl = quantity * best_price

        return pnl + liquidation_pnl

    def position(self):
        return self.user_variables["inventory"] * self.midprice(), self.user_variables["cash"]

    def step(self, action=None):
        transactions = []

        if action is not None:
            bid = Order(action[0], action[1], 'bid')
            ask = Order(action[2], action[3], 'ask')
            self.set_order(bid, ask)

        if self.market_variables['timestep'] + self.dt >= self.next_event:
            self.market_variables['events'].append(self.next_event)
            self.previous_event = self.next_event
            orders = self._sample_orders(self.midprice())
            self.next_event = self.event_process.sample(self.market_variables['timestep'], self.dt,
                                                        np.array(self.market_variables['events']))
            np.random.shuffle(orders)
            for order in orders:
                transactions += self.lob.send_order(order)

            self.market_variables['timestep'] = self.next_event
        else:
            self.market_variables['timestep'] += self.dt

        self.market_variables['midprice'].append(self.midprice())
        self.market_variables['risk_free_rate'] = self.risk_free_process.sample(self.market_variables['risk_free_rate'],
                                                                                self.dt)
        self.market_variables['spread'] = self.spread_process.sample(
            self.market_variables['spread'], self.dt)
        self.ask_process.mean = self.market_variables['risk_free_rate'] + \
            self.market_variables['spread'] / 2
        self.bid_process.mean = self.market_variables['risk_free_rate'] - \
            self.market_variables['spread'] / 2

        transaction_pnl = 0
        if transactions:
            transaction_pnl, delta_inventory = self._calculate_pnl(
                transactions)
            self.user_variables["cash"] += transaction_pnl
            self.user_variables["inventory"] += delta_inventory

        return transactions, self.position(), transaction_pnl

    @property
    def market_timestep(self):
        return self.market_variables['timestep']


if __name__ == "__main__":
    simulator = MarketSimulator(100, spread_mean=.1)
    simulator.fill(10)

    simulator.lob.plot()
