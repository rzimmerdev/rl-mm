import os

import matplotlib.pyplot as plt
import numpy as np

from src.agent.agent import PPOAgent
from src.agent.trainer import RLTrainer
from src.env import MarketEnv


def config_reshape(action, size, low, high):
    action = low + (high - low) * action / size
    action = np.clip(action, low, high)
    return action


def train():
    env = MarketEnv()
    precision = int(1e3)
    action_reshape = lambda action: config_reshape(action, precision, env.action_space.low, env.action_space.high)
    state_dim = env.observation_space.shape[0]
    action_dim = [int(precision) for _ in range(env.action_space.shape[0])]
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        policy_hidden_dims=(state_dim * 10, state_dim * 10, state_dim ** 2),
        value_hidden_dims=(state_dim * 10, state_dim, int(np.sqrt(state_dim))),
        action_reshape=action_reshape,
        lr=1e-3,
        gamma=0.99,
        eps_clip=0.2,
        gae_lambda=0.98,
        entropy_coef=0.01
    )

    trainer = RLTrainer(env, agent)
    rewards = trainer.train(num_episodes=int(1e4))

    plot_rewards(rewards)


def plot_rewards(reward_history, plot_dir="plots"):
    plt.figure(figsize=(12, 6))
    # remove outliers
    reward_history = [r for r in reward_history if abs(r) < 5000]
    plt.plot(reward_history, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('PPO Training Progress')
    plt.legend()
    plt.grid(True)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.savefig(f'{plot_dir}/reward_plot.png')
    plt.show()


if __name__ == '__main__':
    train()
