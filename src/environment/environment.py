class BaseEnvironment:
    def __init__(self):
        pass

    @property
    def state(self):
        return None

    def step(self, action):
        pass

    def reset(self):
        pass


class SamplingEnvironment(BaseEnvironment):
    def __init__(self):
        self.simulator = simulator

    @property
    def state(self):
        return self.simulator.state

    def step(self, action):
        return self.simulator.step(action)

    def reset(self):
        return self.simulator.reset()