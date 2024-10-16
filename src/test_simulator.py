import numpy as np

from src.framework import Agent, RewardSpace
from src.simulator.environment import LobEnvironment, MarketInterfaceMock, LobActionSpace
from src.simulator.simulator import Simulator


class Rewards(RewardSpace):
    def contains(self, action) -> bool:
        return True


def main():
    class PPOAgent(Agent):
        def act(self, state: np.array) -> np.array:
            #
            return np.array([np.inf, 0, -np.inf, 0])

        def update(self, next_state, reward, done):
            pass

    agent = PPOAgent()
    api = MarketInterfaceMock()
    environment = LobEnvironment(api, LobActionSpace(), Rewards(), market_time=100)

    simulator = Simulator(agent, environment)

    simulator.run()
    print("Simulation complete")
    print("Final state:", environment.state)


if __name__ == "__main__":
    main()
