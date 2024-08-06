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
    def __init__(self, simulator: SimpleSimulator = None, interval=1, depth=25):
        self.dt_mean = 10
        self.spread_mean = 0.1
        self.size_mean = 10
        self.interval = interval
        self.depth = depth
        self.simulator = simulator or SimpleSimulator(self.dt_mean, self.spread_mean, self.size_mean)
        self.spread_velocity = 0
        self.spread_acceleration = 0

    @property
    def lob(self):
        return self.simulator.lob

    def state(self):
        return (len(self.lob.bids.inorder()[:self.depth]) - len(self.lob.asks.inorder()[:self.depth]) / self.depth,
                self.lob.mid_price,
                self.lob.spread,
                self.spread_velocity,
                self.spread_acceleration)

    def step(self, action):
        self.simulator.run(self.interval)
        # update spread velocity and acceleration
        # average over last 5 spread values
        self.spread_velocity = 4 / 5 * self.spread_velocity + 1 / 5 * (self.simulator.lob.spread - self.lob.spread)
        self.spread_acceleration = 4 / 5 * self.spread_acceleration + 1 / 5 * (self.spread_velocity - self.spread_velocity)
