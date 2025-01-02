import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from src.agent.model import PolicyNetwork, ValueNetwork
from src.agent.replay_buffer import ReplayBuffer


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, eps_clip=0.2, gae_lambda=0.95, entropy_coef=0.01):
        self.policy_network = PolicyNetwork(state_dim, action_dim).cuda()
        self.value_network = ValueNetwork(state_dim).cuda()
        self.optimizer = optim.Adam([
            {'params': self.policy_network.parameters()},
            {'params': self.value_network.parameters()}
        ], lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef

    def compute_gae(self, rewards, values, dones):
        advantages = []
        advantage = 0
        values = values.tolist()
        values.append(0)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.gae_lambda * advantage * (1 - dones[t])
            advantages.insert(0, advantage)
        return advantages, [adv + val for adv, val in zip(advantages, values[:-1])]

    def collect_trajectories(self, env, rollout_length=1000):
        trajectories = ReplayBuffer()
        state, _ = env.reset()
        for _ in range(rollout_length):
            action, dist, log_prob = self.policy_network.act(torch.tensor(state, dtype=torch.float32).cuda())
            next_state, reward, done, trunc, _ = env.step(
                action.item() if isinstance(action, torch.Tensor) else [a.item() for a in action])
            done = done or trunc
            trajectories.add(state, action, log_prob, reward, done)
            state = next_state
            if done:
                break
        return trajectories

    def update(self, trajectories, epochs=10, batch_size=64):
        states = torch.tensor(np.array(trajectories.states), dtype=torch.float32).cuda()
        actions = torch.tensor(trajectories.actions).cuda()
        old_log_probs = torch.tensor(trajectories.log_probs, dtype=torch.float32).cuda()
        rewards = torch.tensor(trajectories.rewards, dtype=torch.float32).cuda()
        dones = torch.tensor(trajectories.dones, dtype=torch.float32).cuda()

        state_values = self.value_network(states).squeeze().detach().cpu().numpy()
        advantages, returns = self.compute_gae(rewards.cpu().numpy(), state_values, dones.cpu().numpy())

        advantages = torch.tensor(advantages, dtype=torch.float32).cuda()
        returns = torch.tensor(returns, dtype=torch.float32).cuda()

        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, advantages, returns)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):  # Number of training epochs
            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in data_loader:
                action, dist, log_probs = self.policy_network.act(batch_states)
                entropy = dist.entropy().mean()

                ratios = torch.exp(log_probs - batch_old_log_probs)

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                state_values = self.value_network(batch_states).squeeze()
                value_loss = nn.MSELoss()(state_values, batch_returns)

                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
