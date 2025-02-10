import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
GAMMA = 0.99
LAMBDA = 0.98
EPS_CLIP = 0.2
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 100
TRAJECTORY_LEN = 1024
EPISODES = int(1e3)

# Environment setup
env = gym.make("CartPole-v1")
state_shape = env.observation_space.shape


def preprocess_state(state):
    return state


# Actor-Critic Networks
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, padding="same"),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, padding="same"),
            nn.MaxPool2d(kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.MaxPool2d(kernel_size=4),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        return self.conv(x)


class Actor(nn.Module):
    def __init__(self, input_shape):
        super(Actor, self).__init__()
        self.cnn = CNNFeatureExtractor()
        feature_size = state_shape[0]
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.output = nn.Linear(256, 2)  # Steering distribution

    def forward(self, state):
        x = self.fc(state)
        return nn.Softmax(dim=-1)(self.output(x))


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        feature_size = state_shape[0]
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.fc(state)


# Replay Buffer
class ReplayBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards, self.dones, self.log_probs = [], [], [], [], []
        self.values = []

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self, gamma, lam):
        advantages, returns = [], []
        gae = 0
        next_value = 0
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
            next_value = self.values[t]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def clear(self):
        self.states, self.actions, self.rewards, self.dones, self.log_probs, self.values = [], [], [], [], [], []


# Training loop
def train():
    actor = Actor(state_shape)
    critic = Critic()
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE)
    buffer = ReplayBuffer()

    episode_rewards = []

    for episode in range(EPISODES):
        render = (episode % 5 == 0)  # Render every 5th episode
        state, _ = env.reset()
        state = preprocess_state(state)

        rewards = 0

        for t in range(TRAJECTORY_LEN):
            if render:
                env.render()  # Render the environment
            state_tensor = torch.tensor(state, dtype=torch.float32)
            probs = actor(state_tensor)
            dist = Categorical(probs)

            action = dist.sample()
            log_probs = dist.log_prob(action)

            value = critic(state_tensor).item()

            next_state, reward, done, _, _ = env.step(action.item())
            next_state = preprocess_state(next_state)

            buffer.store(state, action.item(), reward, done, log_probs, value)

            state = next_state
            if done:
                break
            rewards += reward

        advantages, returns = buffer.compute_gae(GAMMA, LAMBDA)
        states = torch.tensor(np.array(buffer.states), dtype=torch.float32)
        actions = torch.tensor(np.array(buffer.actions), dtype=torch.float32)
        log_probs_old = torch.tensor(buffer.log_probs, dtype=torch.float32)

        episode_reward = sum(buffer.rewards)
        print(f"Episode: {episode + 1}, Reward: {episode_reward}")

        for _ in range(EPOCHS):
            indices = np.random.permutation(len(states))[:BATCH_SIZE]
            sampled_states = states[indices].squeeze(1)
            sampled_actions = actions[indices]
            sampled_log_probs_old = log_probs_old[indices]
            sampled_advantages = advantages[indices]
            sampled_returns = returns[indices]

            # Compute New Policy
            probs = actor(sampled_states)
            dist = Categorical(probs)

            log_probs = dist.log_prob(sampled_actions.long())
            ratio = torch.exp(log_probs - sampled_log_probs_old)

            surr1 = ratio * sampled_advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * sampled_advantages.unsqueeze(1)

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (critic(sampled_states).squeeze() - sampled_returns).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        episode_rewards.append(rewards)
        buffer.clear()

    env.close()

    import matplotlib.pyplot as plt
    # moving average
    episode_rewards = np.array(episode_rewards)
    episode_rewards = np.convolve(episode_rewards, np.ones(100), 'valid') / 100
    plt.plot(range(EPISODES), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()


if __name__ == "__main__":
    train()
