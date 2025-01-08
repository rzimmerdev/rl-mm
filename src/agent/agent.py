import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from src.agent.model import PolicyNetwork, MultiHeadPolicyNetwork, MLP
from src.agent.replay_buffer import ReplayBuffer


def default_reshape(action):
    return action


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        policy_hidden_dims=(128, 128),
        value_hidden_dims=(128, 128),
        action_reshape=default_reshape,
        lr=1e-4,
        gamma=0.99,
        eps_clip=0.2,
        gae_lambda=0.95,
        entropy_coef=0.01
    ):
        self.policy_network = MultiHeadPolicyNetwork(state_dim, action_dim, policy_hidden_dims).cuda() \
            if isinstance(action_dim, list) else PolicyNetwork(state_dim, action_dim).cuda()
        self.value_network = MLP(state_dim, 1, hidden_units=value_hidden_dims).cuda()
        self.optimizer = optim.Adam([
            {'params': self.policy_network.parameters()},
            {'params': self.value_network.parameters()}
        ], lr=lr)
        self.action_reshape = action_reshape
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef

    def compute_gae(self, rewards, values, dones):
        advantages = []
        advantage = 0
        value = values.tolist()
        value = [value] if isinstance(value, float) else value
        value.append(0)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * value[t + 1] * (1 - dones[t]) - value[t]
            advantage = delta + self.gamma * self.gae_lambda * advantage * (1 - dones[t])
            advantages.insert(0, advantage)
        return advantages, [adv + val for adv, val in zip(advantages, value[:-1])]

    def collect_trajectories(self, env, rollout_length=1000):
        trajectories = ReplayBuffer()
        state, _ = env.reset()

        for _ in range(rollout_length):
            action, log_prob, _ = self.policy_network.act(torch.tensor(state, dtype=torch.float32).cuda().unsqueeze(0))
            next_state, reward, done, trunc, _ = env.step(self.action_reshape(self.policy_network.get_action(action)))
            done = done or trunc
            trajectories.add(state, action, log_prob, reward, done)
            state = next_state
            if done:
                break

        return trajectories

    def update(self, trajectories, epochs=20, batch_size=512):
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

        episode_loss = 0

        for epoch in range(epochs):  # Number of training epochs
            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in data_loader:
                _, _, dist = self.policy_network.act(batch_states)
                log_probs = self.logprobs(dist, batch_actions)
                entropy = self.entropy(dist)

                ratios = torch.exp(log_probs.sum(dim=-1) - batch_old_log_probs.sum(dim=-1))

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                state_values = self.value_network(batch_states).squeeze()
                value_loss = nn.MSELoss()(state_values, batch_returns)

                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                episode_loss += loss.item()

        return episode_loss / epochs

    def logprobs(self, dists, actions):
        if isinstance(dists, list):
            return torch.stack([dist.log_prob(act) for dist, act in zip(dists, actions.T)]).T
        return dists.log_prob(actions)

    def entropy(self, dists):
        if isinstance(dists, list):
            return torch.stack([dist.entropy() for dist in dists]).mean()
        return dists.entropy().mean()

    def save_weights(self, path):
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict':  self.value_network.state_dict()
        }, path)

    def load_weights(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])


def test_ppo_agent_multidimensional():
    # Create a multidimensional mock environment
    class MultiDimEnv:
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
            self.state = np.zeros(self.observation_space.shape)

        def reset(self):
            self.state = np.random.uniform(-1, 1, self.observation_space.shape)
            return self.state, {}

        def step(self, action):
            next_state = np.random.uniform(-1, 1, self.observation_space.shape)
            reward = -np.sum((self.state - next_state) ** 2)
            done = np.random.rand() > 0.9  # Randomly terminate the episode
            trunc = False
            return next_state, reward, done, trunc, {}

    env = MultiDimEnv()

    # Initialize the agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, [int(1e4) for _ in range(action_dim)])

    # Collect trajectories
    trajectories = agent.collect_trajectories(env, rollout_length=100)

    # Perform an update
    agent.update(trajectories, epochs=1, batch_size=32)
    print("Update method executed without errors for multidimensional environment.")


# Call the test function
if __name__ == "__main__":
    test_ppo_agent_multidimensional()
