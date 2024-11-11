import matplotlib
import numpy as np
import plotly.graph_objects as go

from src.agents.avstoikov import AvellanedaStoikov
from src.agents.ppo_agent import PPOAgent
from src.framework import Agent, RewardSpace
from src.simulator.environment import LobEnvironment, MarketInterfaceMock, LobActionSpace
from src.simulator.simulator import Simulator


class Rewards(RewardSpace):
    def contains(self, action) -> bool:
        return True




def main():
    agent = AvellanedaStoikov()
    market_dt = 1e-1
    market_time = 1e2 * 5
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
    # quotes[:, 1] is market time in ms, plot in seconds
    fig = go.Figure()

    # Add market quotes line
    fig.add_trace(go.Scatter(
        x=environment.quotes[100:, 0],
        y=environment.quotes[100:, 1],
        mode='lines',
        name='Market Quotes'
    ))

    # Add agent ask and bid quotes with smaller markers
    fig.add_trace(go.Scatter(
        x=environment.agent_quotes[100:, 0],
        y=environment.agent_quotes[100:, 1],
        mode='markers',
        marker=dict(color='red', size=5),
        name='Agent Ask Quotes'
    ))

    fig.add_trace(go.Scatter(
        x=environment.agent_quotes[100:, 0],
        y=environment.agent_quotes[100:, 2],
        mode='markers',
        marker=dict(color='green', size=5),
        name='Agent Bid Quotes'
    ))

    # Show the figure
    fig.show()


if __name__ == "__main__":
    main()
