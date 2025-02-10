import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader

# Hyperparameters
GAMMA = 0.99
LAMBDA = 0.98
EPS_CLIP = 0.2
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 100
TRAJECTORY_LEN = 1024
EPISODES = int(1e3)
PRECISION = 100  # Precision level actions

# Environment setup
env = gym.make("LunarLander-v3", continuous=True)
state_shape = env.observation_space.shape
action_shape = env.action_space.shape


def preprocess_state(state):
    return state


def preprocess_action(action):
    return action / PRECISION * 2 - 1


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
    def __init__(self, ):
        super(Actor, self).__init__()
        feature_size = state_shape[0]
        output_dims = action_shape[0]
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.output = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(256, PRECISION),
                    nn.Softmax(dim=-1)
                )
                for _ in range(output_dims)
            ]
        )

    def forward(self, state):
        x = self.fc(state)
        out = [layer(x) for layer in self.output]
        return out


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
        self.states, self.actions, self.rewards, self.dones, self.log_probs, self.values = [], [], [], [], [], []
        self.gamma = 0.99
        self.lam = 0.98

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    # def compute_gae(self, rewards, values, dones):
    #     advantages, returns = [], []
    #     gae = 0
    #     next_value = 0
    #     for t in reversed(range(len(self.rewards))):
    #         delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
    #         gae = delta + gamma * lam * (1 - self.dones[t]) * gae
    #         advantages.insert(0, gae)
    #         returns.insert(0, gae + self.values[t])
    #         next_value = self.values[t]
    #     return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)
    #

    def compute_gae(self, rewards, values, dones):
        advantages = []
        advantage = 0
        _values = values.tolist()
        _values.append(0)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * _values[t + 1] * (1 - dones[t]) - _values[t]
            advantage = delta + self.gamma * self.lam * advantage * (1 - dones[t])
            advantages.insert(0, advantage)
        return advantages, [adv + val for adv, val in zip(advantages, _values[:-1])]

    def clear(self):
        self.states, self.actions, self.rewards, self.dones, self.log_probs, self.values = [], [], [], [], [], []


# Training loop
def train():
    actor = Actor()
    critic = Critic()
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE)
    buffer = ReplayBuffer()

    for episode in range(EPISODES):
        render = (episode % 5 == 0)  # Render every 5th episode
        state, _ = env.reset()
        state = preprocess_state(state)

        for t in range(TRAJECTORY_LEN):
            if render:
                env.render()  # Render the environment

            state_tensor = torch.tensor(state, dtype=torch.float32)

            probs = actor(state_tensor)
            value = critic(state_tensor).item()

            dists = [Categorical(prob) for prob in probs]
            action = [dist.sample() for dist in dists]
            log_probs = torch.stack([dist.log_prob(act) for dist, act in zip(dists, action)])

            action = np.array([
                act.item() for act in action
            ])

            next_state, reward, done, trunc, _ = env.step(
                np.array([
                    preprocess_action(act) for act in action
                ])
            )
            next_state = preprocess_state(next_state)

            buffer.store(state, action, reward, done, log_probs, value)

            state = next_state
            if done or trunc:
                break

        episode_reward = sum(buffer.rewards)
        print(f'Episode {episode}, Reward: {episode_reward}')

        states = torch.tensor(np.array(buffer.states), dtype=torch.float32)
        actions = torch.tensor(np.array(buffer.actions), dtype=torch.float32)
        log_probs_old = torch.stack(buffer.log_probs).detach()

        values = critic(states).squeeze().detach().numpy()
        advantages, returns = buffer.compute_gae(buffer.rewards, values, buffer.dones)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        dataset = TensorDataset(states, actions, log_probs_old, advantages, returns)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


        # Training loop
        for _ in range(EPOCHS):
            for sampled_states, sampled_actions, sampled_log_probs_old, sampled_advantages, sampled_returns in data_loader:
                probs = actor(sampled_states)
                state_values = critic(sampled_states).squeeze()

                dists = [Categorical(prob) for prob in probs]

                log_probs = torch.stack([dist.log_prob(act) for dist, act in zip(dists, sampled_actions.T)]).T
                ratio = torch.exp(log_probs.sum(dim=-1) - sampled_log_probs_old.sum(dim=-1))

                surr1 = ratio * sampled_advantages
                surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * sampled_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(state_values, sampled_returns)

                loss = actor_loss + 0.5 * critic_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        buffer.clear()

    env.close()


if __name__ == "__main__":
    train()
