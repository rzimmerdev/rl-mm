
import numpy as np

from src.agent.agent import PPOAgent
from src.agent.trainer import RLTrainer
from src.env import MarketEnv


def config_reshape(action, size, low, high):
    action = low + (high - low) * action / size
    action = np.clip(action, low, high)
    return action


def train():
    env = MarketEnv(spread_mean=250)
    precision = int(1e3)
    action_reshape = lambda action: config_reshape(action, precision, env.action_space.low, env.action_space.high / 1000)
    state_dim = env.observation_space.shape[0]
    action_dim = [int(precision) for _ in range(env.action_space.shape[0])]
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        policy_hidden_dims=(state_dim * 5, state_dim * 10, state_dim * 10),
        value_hidden_dims=(state_dim * 10, state_dim, int(np.sqrt(state_dim))),
        action_reshape=action_reshape,
        lr=1e-3,
        gamma=0.99,
        eps_clip=0.2,
        gae_lambda=0.99,
        entropy_coef=0.02
    )

    trainer = RLTrainer(env, agent)
    trainer.train(num_episodes=int(1e4))


if __name__ == '__main__':
    train()
