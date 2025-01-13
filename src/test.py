import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
from env import MarketEnv
from src.agent import PPOAgent, RLTrainer


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
    trainer = RLTrainer(env, agent)
    agent = trainer.load()

    # Simulate multiple runs
    num_simulations = 20
    all_values = []

    for _ in range(num_simulations):
        value = [np.sum(env.simulator.position())]
        state, _ = env.reset()

        while not env.done:
            action, log_prob, _ = agent.policy_network.act(
                torch.tensor(state, dtype=torch.float32).cuda().unsqueeze(0))
            action = action_reshape(agent.policy_network.get_action(action))
            state, reward, done, trunc, _ = env.step(action)
            value.append(np.sum(env.simulator.position()))

        value = np.array(value)
        all_values.append(value)

    # Align arrays (pad shorter arrays with NaN)
    max_length = max(len(run) for run in all_values)
    aligned_values = np.full((len(all_values), max_length), np.nan)

    for i, run in enumerate(all_values):
        aligned_values[i, :len(run)] = run

    # Calculate mean and std deviation
    mean_returns = np.nanmean(aligned_values, axis=0)
    std_returns = np.nanstd(aligned_values, axis=0)

    mean_returns = mean_returns[::5]
    std_returns = std_returns[::5]
    max_length = len(mean_returns)
    ts = range(max_length)
    # multiply by 5
    ts = [x * 5 for x in ts]

    plt.figure(figsize=(10, 6))
    plt.plot(ts, mean_returns, label="Mean Returns", color='blue')
    plt.fill_between(
        range(max_length),
        mean_returns - std_returns,
        mean_returns + std_returns,
        color='blue',
        alpha=0.2,
        label="±1 Std Dev"
    )
    plt.xlabel("Time Step")
    plt.ylabel("Returns")
    plt.title("Mean Returns with ±1 Std Dev")
    plt.legend()
    plt.grid()
    plt.show()

    # value = [np.sum(env.simulator.position())]
    # t = [0]
    #
    # state, _ = env.reset()
    #
    # while not env.done:
    #     action, log_prob, _ = agent.policy_network.act(torch.tensor(state, dtype=torch.float32).cuda().unsqueeze(0))
    #     next_state, reward, done, trunc, _ = env.step(action_reshape(agent.policy_network.get_action(action)))
    #     value.append(np.sum(env.simulator.position()))
    #     t.append(env.simulator.market_timestep)
    #
    # value = np.array(value)
    # t = np.array(t)
    #
    # # remove elements up to the first non-zero return
    # idx = np.argmax(value != 0)
    # value = value[idx:]
    # t = t[idx:]
    #
    # # Take elements spaced every cumulative increase of 5 units in t
    # selected_indices = [0]  # Start with the first element
    # cumulative_time = t[0]
    #
    # for i in range(1, len(t)):
    #     if t[i] - cumulative_time >= 5:
    #         selected_indices.append(i)
    #         cumulative_time = t[i]
    #
    # # Extract the selected values and times
    # value = value[selected_indices]
    # t = t[selected_indices]
    #
    # plot_financial_return(value, t)


def plot_financial_return(value, t):
    plt.figure(figsize=(12, 6))
    plt.plot(t, value, label='Return')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.title('Financial Return')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    test()
