import matplotlib.pyplot as plt
import numpy as np
import torch

from env import MarketEnv
from src.agent.agent import PPOAgent


def config_reshape(action, size, low, high):
    # range action is 0 to size, required is low to high
    return low + (high - low) * action / size


def test():
    env = MarketEnv()
    precision = int(1e3)
    action_reshape = lambda action: config_reshape(action, precision, env.action_space.low, env.action_space.high)
    state_dim = env.observation_space.shape[0]
    action_dim = [int(precision) for _ in range(env.action_space.shape[0])]
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        policy_hidden_dims=(state_dim * 5, state_dim * 10, state_dim * 10),
        value_hidden_dims=(state_dim * 10, state_dim, int(np.sqrt(state_dim))),
        action_reshape=action_reshape,
        lr=1e-4,
        gamma=0.99,
        eps_clip=0.2,
        gae_lambda=0.98,
        entropy_coef=0.02
    )

    # try to load weights if path exists
    try:
        agent.load_weights('agent_weights.pth')
    except FileNotFoundError:
        print('No weights found')

    value = [np.sum(env.simulator.position())]

    state, _ = env.reset()

    while not env.done:
        action, log_prob, _ = agent.policy_network.act(torch.tensor(state, dtype=torch.float32).cuda().unsqueeze(0))
        next_state, reward, done, trunc, _ = env.step(action_reshape(agent.policy_network.get_action(action)))
        value.append(np.sum(env.simulator.position()))

    plot_financial_return(value)


def plot_financial_return(value):
    plt.figure(figsize=(12, 6))
    plt.plot(value, label='Financial Return')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.title('Financial Return')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    test()
