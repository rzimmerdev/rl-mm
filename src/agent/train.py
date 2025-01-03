import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from src.agent.agent import PPOAgent
from src.agent.trainer import RLTrainer


def config_reshape(action, size, low, high):
    # range action is 0 to size, required is low to high
    return low + (high - low) * action / size


def train():
    env = gym.make("Ant-v5", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, "videos/")
    precision = int(1e3)
    action_reshape = lambda action: config_reshape(action, precision, env.action_space.low, env.action_space.high)
    state_dim = env.observation_space.shape[0]
    action_dim = [int(precision) for _ in range(env.action_space.shape[0])]
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        policy_hidden_dims=(state_dim * 10, state_dim, int(np.sqrt(10 * state_dim))),
        value_hidden_dims=(state_dim * 10, state_dim, int(np.sqrt(10 * state_dim))),
        action_reshape=action_reshape,
        lr=1e-4,
        gamma=0.99,
        eps_clip=0.2,
        gae_lambda=0.99,
        entropy_coef=0.001
    )

    trainer = RLTrainer(env, agent)
    rewards = trainer.train(num_episodes=int(1e4))

    env.close()
    plot_rewards(rewards)


def plot_rewards(reward_history):
    plt.figure(figsize=(12, 6))
    plt.plot(reward_history, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('PPO Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/reward_plot.png')
    plt.show()


if __name__ == '__main__':
    train()
