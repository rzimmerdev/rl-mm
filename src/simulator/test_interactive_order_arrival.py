import numpy as np
from tqdm import tqdm


class EventSampler:
    def __init__(self, dt=1):
        self.t = 0
        self.next_event = 0
        self.prev_event = 0
        self.events = []
        self.agent_events = []
        self.dt = dt

    def step(self):
        if self.prev_event + self.next_event < self.t:
            self.t += self.next_event
            self.prev_event = self.t
            self.events.append(self.next_event)
            self.next_event = np.random.exponential(1)
            return

        if np.random.uniform() < 1e-2 * self.dt:
            self.prev_event = self.t
            self.events.append(self.t)
            self.agent_events.append(self.t)
            self.next_event = np.random.exponential(1)

        self.t += self.dt

    def run(self, market_close, use_tqdm=True):
        self.t = 0
        self.next_event = np.random.exponential(1)
        self.events = []

        # use tqdm
        if use_tqdm:
            with tqdm(total=market_close) as pbar:
                while self.t < market_close:
                    self.step()
                    pbar.update(self.dt)

        # don't use tqdm
        else:
            while self.t < market_close:
                self.step()

        return self.events
