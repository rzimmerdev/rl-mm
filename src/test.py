import os
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from env import MarketEnv
from src.agent import PPOAgent, RLTrainer
from src.agent.closed import TestStoikov


def config_reshape(action, size, low, high):
    return low + (high - low) * action / size



def returns(ppnl):
    # get pct change returns
    _pnl = np.array(ppnl) + 100
    pct_change = np.diff(_pnl) / _pnl[:-1]
    # remove nan and invalid values (inf, -inf)
    pct_change = pct_change[~np.isnan(pct_change)]
    pct_change = pct_change[~np.isinf(pct_change)]
    return pct_change

precision = int(1e3)


def test():
    num_simulations = 20
    n_levels = 10
    rf_mean = -0.02
    env = MarketEnv(n_levels, risk_free_mean=rf_mean)
    action_reshape = lambda action: config_reshape(action, precision, env.action_space.low,
                                                   env.action_space.high / precision)
    state_dim = env.observation_space.shape[0]
    action_dim = [int(precision) for _ in range(env.action_space.shape[0])]
    agent = PPOAgent(
        num_features=state_dim - 4 * 10,
        num_depth=10,
        action_dim=action_dim,
        policy_hidden_dims=(state_dim * 24, state_dim * 10, state_dim * 4),
        value_hidden_dims=(state_dim * 10, state_dim * 5, state_dim // 2),
        action_reshape=action_reshape,
        attention_heads=8,
        gamma=0.99,
        eps_clip=0.2,
        gae_lambda=0.98,
        entropy_coef=0.1
    )

    trainer = RLTrainer(env, agent)

    agent = trainer.load()

    log_dir = trainer.latest_path
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")][-1]
    event_acc = EventAccumulator(event_files)
    event_acc.Reload()
    tags = event_acc.Tags()['scalars']

    metrics = pd.DataFrame()

    for tag in tags:
        event_acc.Reload()
        data = pd.DataFrame(event_acc.Scalars(tag))
        data = data.set_index('step')
        metrics[tag] = data['value']

    smoothing = 0.999
    rewards = metrics['reward'].ewm(alpha=1 - smoothing).mean()[200:]
    linear_trend = np.polyfit(rewards.index, rewards, 1)
    linear_trend = np.poly1d(linear_trend)
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Reward')
    plt.plot(rewards.index, linear_trend(rewards.index), label='Trend', linestyle='--')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward vs Episode')
    plt.legend()
    plt.show()

    # moving average critic and actor loss
    # remove outliers (3 std)
    critic_loss = metrics['critic_loss']
    actor_loss = metrics['actor_loss']

    # Remove outliers beyond 3 standard deviations
    critic_loss = critic_loss[critic_loss < 0.2]
    actor_loss = actor_loss[0.09 > actor_loss]
    actor_loss = actor_loss[200:]
    critic_loss = critic_loss[200:]

    # Apply exponential weighted moving average smoothing
    critic_loss = critic_loss.ewm(alpha=1 - smoothing).mean()
    actor_loss = actor_loss.ewm(alpha=1 - smoothing).mean()

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Critic loss plot with log scale
    ax[0].plot(critic_loss, label='Critic Loss')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Critic Loss vs Episode')
    ax[0].legend()

    # Actor loss plot with log scale
    ax[1].plot(actor_loss, label='Actor Loss')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Actor Loss vs Episode')
    ax[1].legend()

    plt.show()

    all_pnl = []

    times = {
        "episode": [],
        "actor": [],
        "critic": []
    }

    test_stoikov = TestStoikov()

    all_pnl, all_returns, times = run_agent_simulation(agent, env, num_simulations)
    stoikov_pnl, stoikov_returns = run_stoikov_simulation(env, num_simulations, test_stoikov)
    long_pnl, long_returns = run_long_only_simulation(env, num_simulations)

    for key, value in times.items():
        mean = np.mean(value)
        std = np.std(value)
        print(f"{key} mean: {mean}, std: {std}")

    max_length = max(len(run) for run in all_pnl)
    aligned_values = np.full((len(all_pnl), max_length), np.nan)

    for i, run in enumerate(all_pnl):
        aligned_values[i, :len(run)] = run

    mean_returns = np.nanmean(aligned_values, axis=0)
    std_returns = np.nanstd(aligned_values, axis=0)

    mean_returns = mean_returns[::5]
    std_returns = std_returns[::5]
    ts = range(len(mean_returns))
    ts = [x * 5 for x in ts]

    plt.figure(figsize=(10, 6))
    plt.plot(ts, mean_returns, label="Mean Returns", color='blue')
    plt.fill_between(
        ts, mean_returns - std_returns, mean_returns + std_returns,
        color='blue', alpha=0.2, label="±1 Std Dev"
    )

    plt.xlabel("Time Step")
    plt.ylabel("Returns")
    plt.title("Mean Returns with ±1 Std Dev")
    plt.legend()
    plt.grid()
    plt.show()

    # Assuming all_returns and stoikov_returns are lists of PnL data for each simulation
    all_returns = np.array([pnl[-1] for pnl in all_returns])
    std_returns = np.std(all_returns)
    mean_returns = np.mean(all_returns)

    stoikov_returns = np.array([pnl[-1] for pnl in stoikov_returns])
    stoikov_std_returns = np.std(stoikov_returns)
    stoikov_mean_returns = np.mean(stoikov_returns)

    long_returns = np.array([pnl[-1] for pnl in long_returns])
    long_std_returns = np.std(long_returns)
    long_mean_returns = np.mean(long_returns)

    # Risk-free rate for Sortino (adjusted for daily returns in a trading environment)
    rf = rf_mean / 1 / 252 / 6.5 / 60

    # Function to calculate downside deviation
    def downside_deviation(pnls):
        # Filter out positive returns, keeping only the negative ones
        negative_returns = [pnl for pnl in pnls if pnl < 0]
        return np.std(negative_returns) if negative_returns else 0

    # Sortino ratio calculation
    def sortino(mu, downside_std):
        return (mu - rf) / downside_std if downside_std != 0 else np.nan

    # Calculate downside deviation for each dataset
    downside_std_returns = downside_deviation(all_returns)
    downside_std_stoikov_returns = downside_deviation(stoikov_returns)
    downside_std_long_returns = downside_deviation(long_returns)

    # Print results using Sortino ratio
    print(
        f"Mean PnL: {mean_returns}, Std PnL: {std_returns}, Sortino Ratio: {sortino(mean_returns, downside_std_returns)}")
    print(
        f"Mean Stoikov PnL: {stoikov_mean_returns}, Std Stoikov PnL: {stoikov_std_returns}, Sortino Ratio: {sortino(stoikov_mean_returns, downside_std_stoikov_returns)}")
    print(
        f"Mean Long PnL: {long_mean_returns}, Std Long PnL: {long_std_returns}, Sortino Ratio: {sortino(long_mean_returns, downside_std_long_returns)}")


def run_agent_simulation(agent, env, num_simulations):
    all_pnl, all_returns = [], []
    times = {"actor": [], "critic": [], "episode": []}

    for _ in range(num_simulations):
        _pnl = [0]
        state, _ = env.reset()
        start = time.time()

        while not env.done:
            actor = time.time()
            action, log_prob, _ = agent.policy_network.act(
                torch.tensor(state, dtype=torch.float32).cuda().unsqueeze(0))
            times["actor"].append(time.time() - actor)

            critic = time.time()
            agent.value_network(torch.tensor(state, dtype=torch.float32).cuda().unsqueeze(0))
            times["critic"].append(time.time() - critic)

            action = agent.action_reshape(agent.policy_network.get_action(action))
            state, reward, done, trunc, _ = env.step(action)

            _pnl.append(np.sum(env.simulator.position()))

        all_pnl.append(_pnl)
        all_returns.append(returns(_pnl))
        times["episode"].append(time.time() - start - times["critic"][-1])

    return all_pnl, all_returns, times

def run_stoikov_simulation(env, num_simulations, test_stoikov):
    stoikov_pnl, stoikov_returns = [], []

    for _ in range(num_simulations):
        _stoi_pnl = [0]
        state, _ = env.reset()

        while not env.done:
            midprice = env.simulator.midprice()
            spread = env.simulator.spread()
            test_stoikov.observe(midprice, spread)
            state, reward, done, trunc, _ = env.step(None)  # No agent actions
            _stoi_pnl.append(test_stoikov.pnl())

        stoikov_pnl.append(_stoi_pnl)
        stoikov_returns.append(returns(_stoi_pnl))
        test_stoikov.reset()

    return stoikov_pnl, stoikov_returns

def run_long_only_simulation(env, num_simulations):
    long_pnl, long_returns = [], []

    for _ in range(num_simulations):
        _long_pnl = [100]
        state, _ = env.reset()

        while not env.done:
            midprice = env.simulator.midprice()
            state, reward, done, trunc, _ = env.step(None)  # No agent actions
            _long_pnl.append(midprice)

        long_pnl.append(_long_pnl)
        long_returns.append(returns(_long_pnl))

    return long_pnl, long_returns


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
