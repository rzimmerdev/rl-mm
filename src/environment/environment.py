from simulator import SimpleSimulator


class Environment:
    def __init__(self):
        pass

    @property
    def state(self):
        return None

    def step(self, action):
        pass

    def reset(self):
        pass


class SimpleEnvironment:
    def __init__(self, simulator: SimpleSimulator = None, interval=1):
        self.dt_mean = 10
        self.spread_mean = 0.1
        self.size_mean = 10
        self.interval = 1
        self.simulator = simulator or SimpleSimulator(self.dt_mean, self.spread_mean, self.size_mean)

    def state(self):
        # state R^7 == (order imbalance, mid price, spread, spread slope, spread curvature, spread velocity, spread acceleration)
        # to calculate order imbalance, we need to know the number of buy and sell orders
        # we can use the order book to calculate the order imbalance
        # we can use the mid price to calculate the spread
        # we can use the spread to calculate the spread slope, curvature, velocity, and acceleration
        order_imbalance = 0
        for order in self.simulator.orders:
            order_imbalance += order.quantity if order.side == 'BUY' else -order.quantity

    def step(self):
        self.simulator.run(self.interval)

    def state(self):
        return self.simulator.state