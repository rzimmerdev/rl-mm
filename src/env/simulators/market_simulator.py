import numpy as np

from env.lob import LimitOrderBook, Order

from ..dynamics import GeometricBrownianMotion, CoxIngersollRoss, OrnsteinUhlenbeck, Hawkes


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
    ):
        self.dt = dt
        self.risk_free_mean = risk_free_mean
        self.risk_free_std = risk_free_std
        self.spread_mean = spread_mean
        self.spread_std = spread_std

        bid_center = risk_free_mean - spread_mean / 2
        ask_center = risk_free_mean + spread_mean / 2

        self.bid_process = GeometricBrownianMotion(bid_center, volatility)
        self.ask_process = GeometricBrownianMotion(ask_center, volatility)

        self.risk_free_process = CoxIngersollRoss(risk_free_reversion, risk_free_mean, risk_free_std)
        self.spread_process = OrnsteinUhlenbeck(spread_reversion, spread_mean, spread_std)

        self.event_process = Hawkes(base_event_intensity, branching_ratio=1e-1, decay=0.3)

        self.quantity_distribution = np.random.poisson
        self.event_size_distribution = np.random.poisson
        self.event_size_mean = event_size_mean

        self.tick_size = 6
        self.lob = LimitOrderBook(self.tick_size)

        self.starting_value = starting_value
        self.risk_free = risk_free_mean
        self.spread = spread_mean

        self.market_variables = {
            "last_transaction": self.starting_value,
            "midprice": self.starting_value,
            "risk_free_rate": self.risk_free,
            "spread": self.spread,
            "timestep": 0,
            "events": []
        }

        self.next_event = self.event_process.sample(0, self.dt, np.array(self.market_variables['events']))
        self.previous_event = 0

    def reset(self):
        self.bid_process.mean = self.risk_free_mean - self.spread_mean / 2
        self.ask_process.mean = self.risk_free_mean + self.spread_mean / 2
        self.lob = LimitOrderBook(self.tick_size)
        self.market_variables = {
            "last_transaction": self.starting_value,
            "midprice": self.starting_value,
            "risk_free_rate": self.risk_free,
            "spread": self.spread,
            "timestep": 0,
            "events": []
        }
        self.next_event = self.event_process.sample(0, self.dt, np.array([]))

    def midprice(self):
        best_ask = self.lob.asks.bottom().value.price if self.lob.asks.bottom() is not None else None
        best_bid = self.lob.bids.top().value.price if self.lob.bids.top() is not None else None

        if best_ask is None and best_bid is None:
            return self.market_variables['midprice']
        elif best_ask is None:
            return best_bid
        elif best_bid is None:
            return best_ask

        return (best_ask + best_bid) / 2

    def _sample_orders(self, price):
        bid_size = self.event_size_distribution(self.event_size_mean)
        ask_size = self.event_size_distribution(self.event_size_mean)

        bids = self.bid_process.sample(price, self.dt, bid_size)
        asks = self.ask_process.sample(price, self.dt, ask_size)

        bids_quantity = np.array(self.quantity_distribution(size=bid_size) + 1)
        asks_quantity = np.array(self.quantity_distribution(size=ask_size) + 1)

        orders = [Order(price, quantity, 'ask') for price, quantity in zip(asks, asks_quantity)] + \
                 [Order(price, quantity, 'bid') for price, quantity in zip(bids, bids_quantity)]

        return np.array(orders)

    def step(self):
        transactions = []
        if self.market_variables['timestep'] >= self.next_event:
            self.market_variables['events'].append(self.next_event)
            self.previous_event = self.next_event
            orders = self._sample_orders(self.market_variables['last_transaction'])
            self.next_event = self.event_process.sample(self.market_variables['timestep'], self.dt,
                                                        np.array(self.market_variables['events']))
            np.random.shuffle(orders)
            for order in orders:
                transactions += self.lob.send_order(order)

            self.market_variables['timestep'] = self.next_event
        else:
            self.market_variables['timestep'] += self.dt

        self.market_variables['midprice'] = self.midprice()
        self.market_variables['risk_free_rate'] = self.risk_free_process.sample(self.market_variables['risk_free_rate'],
                                                                                self.dt)
        self.market_variables['spread'] = self.spread_process.sample(self.market_variables['spread'], self.dt)
        self.ask_process.mean = self.market_variables['risk_free_rate'] + self.market_variables['spread'] / 2
        self.bid_process.mean = self.market_variables['risk_free_rate'] - self.market_variables['spread'] / 2

        if transactions:
            self.market_variables["last_transaction"] = np.sum([t.price * t.quantity for t in transactions]) / np.sum([t.quantity for t in transactions])
        else:
            self.market_variables["last_transaction"] = self.market_variables['midprice']

        return transactions

    @property
    def market_timestep(self):
        return self.market_variables['timestep']
