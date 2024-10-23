from typing import Union

import numpy as np

from src.framework import Agent, RewardSpace
from src.simulator.environment import LobEnvironment, MarketInterfaceMock, LobActionSpace, LobStateSpace
from src.simulator.simulator import Simulator


class Rewards(RewardSpace):
    def contains(self, action) -> bool:
        return True

class PPOAgent(Agent):
    def __init__(self):
        self.ask_price = None
        self.ask_quantity = None
        self.bid_price = None
        self.bid_quantity = None

    def act(self, state: Union[np.array]) -> np.array:
        if state is None:
            return None
        ask, bid = LobStateSpace().lob_top(state)
        ask_price = np.random.normal(ask[0], 1)
        if not self.ask_price or np.abs(ask_price - self.ask_price) > 10:
            self.ask_price = np.maximum(ask_price, 0)
            self.ask_quantity = 1
        bid_price = np.random.normal(bid[0], 1)
        if not self.bid_price or np.abs(bid_price - self.bid_price) > 10:
            self.bid_price = np.maximum(bid_price, 0)
            self.bid_quantity = 1

        return np.array([self.ask_price, self.ask_quantity, self.bid_price, self.bid_quantity])

    def update(self, next_state, reward, done):
        pass


def main():
    agent = PPOAgent()
    market_dt = 1e-1
    market_time = 50
    api = MarketInterfaceMock(market_dt)
    environment = LobEnvironment(api, LobActionSpace(), Rewards(), market_time)

    simulator = Simulator(agent, environment)
    discounted_return = None

    try:
        final_state, discounted_return = simulator.run()
        print("Simulation complete")
        print("Final state:", environment.state)
        print("Discounted return:", discounted_return)
    except KeyboardInterrupt:
        print("Simulation interrupted")
        print("Final state:", environment.state)
        print("Discounted return:", discounted_return)
        return

    # since we know environment = lob
    lob = environment.api.lob

    # plot lob
    import matplotlib.pyplot as plt

    asks, bids = lob.state()

    # accumulate quantities per side
    ask_prices, ask_quantities = zip(*asks)
    bid_prices, bid_quantities = zip(*bids)

    ask_quantities = np.cumsum(ask_quantities)
    bid_quantities = np.cumsum(bid_quantities)

    # draw area graph
    plt.fill_between(ask_prices, ask_quantities, step='post', alpha=0.5, label='Asks')
    plt.fill_between(bid_prices, bid_quantities, step='post', alpha=0.5, label='Bids')

    plt.legend()
    plt.show()

    # plot quote line graph
    plt.plot(environment.quotes[100:])
    plt.show()


if __name__ == "__main__":
    main()
