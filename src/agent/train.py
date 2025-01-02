import gymnasium as gym
import matplotlib.pyplot as plt
from src.agent.agent import PPOAgent
from src.agent.trainer import RLTrainer


def train():
    env = gym.make('CartPole-v1')
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=1e-4,
        gamma=0.99,
        eps_clip=0.2,
        gae_lambda=0.99,
        entropy_coef=0.001
    )

    trainer = RLTrainer(env, agent)
    rewards = trainer.train(num_episodes=1000)
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
