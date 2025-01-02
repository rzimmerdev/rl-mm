import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Categorical


def _layer_init(layer, std=torch.sqrt(torch.tensor(2.0)), bias_const=0.0):
    """
    Layer initialization helper.
    """
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyNetwork(nn.Module):
    def __init__(self, observation_dim, action_dims):
        """
        Initialize the policy network.

        Args:
        - observation_dim (int): Dimension of the observation space.
        - action_dims (list[int]): List where each entry specifies the number of possible actions for each dimension.
        """
        super().__init__()
        self.shared_layers = nn.Sequential(
            _layer_init(nn.Linear(observation_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        # Separate output layers for each action dimension
        self.output_layers = nn.ModuleList([
            _layer_init(nn.Linear(64, num_actions)) for num_actions in action_dims
        ])

    def forward(self, observations):
        """
        Forward pass to compute logits for each action dimension.

        Args:
        - observations (torch.Tensor): Observations from the environment.

        Returns:
        - action_distributions (list[torch.distributions.Categorical]): List of categorical distributions for each action dimension.
        """
        shared_features = self.shared_layers(observations)
        action_distributions = [
            Categorical(logits=layer(shared_features)) for layer in self.output_layers
        ]
        return action_distributions



class ValueNetwork(nn.Module):
    def __init__(self, observation_dim):
        """
        Initialize the value network.

        Args:
        - observation_dim (int): Dimension of the observation space.
        """
        super().__init__()
        self.shared_layers = nn.Sequential(
            _layer_init(nn.Linear(observation_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.output_layer = _layer_init(nn.Linear(64, 1))

    def forward(self, observations):
        """
        Forward pass to compute value estimates.

        Args:
        - observations (torch.Tensor): Observations from the environment.

        Returns:
        - values (torch.Tensor): Value estimates.
        """
        shared_features = self.shared_layers(observations)
        values = self.output_layer(shared_features).squeeze(-1)
        return values



class PPO:
    def __init__(self, observation_dim, action_dims, lr=1e-1, gamma=0.99, lam=0.95, epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01):
        """
        Initialize PPO algorithm.

        Args:
        - observation_dim (int): Dimension of observation space.
        - action_dims (list[int]): List specifying the number of possible actions per dimension.
        - lr (float): Learning rate.
        - gamma (float): Discount factor.
        - lam (float): Lambda for GAE (Generalized Advantage Estimation).
        - epsilon (float): Clipping epsilon for PPO.
        - value_coeff (float): Coefficient for value loss.
        - entropy_coeff (float): Coefficient for entropy bonus.
        """
        self.policy_net = PolicyNetwork(observation_dim, action_dims)
        self.value_net = ValueNetwork(observation_dim)
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr
        )
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

    def compute_advantages(self, rewards, values, dones):
        """
        Compute advantages using GAE.

        Args:
        - rewards (torch.Tensor): Collected rewards.
        - values (torch.Tensor): Value estimates.
        - dones (torch.Tensor): Episode termination flags.

        Returns:
        - advantages (torch.Tensor): Computed advantages.
        """
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages

    def update(self, observations, actions, log_probs_old, rewards, dones, epochs=10, batch_size=64):
        """
        Update policy and value networks using PPO loss, performing multiple steps for each trajectory.

        Args:
        - observations (torch.Tensor): Batch of observations.
        - actions (list[torch.Tensor]): Batch of actions for each dimension.
        - log_probs_old (list[torch.Tensor]): Log probabilities from old policy.
        - rewards (torch.Tensor): Batch of rewards.
        - dones (torch.Tensor): Batch of termination flags.
        - epochs (int): Number of update epochs to perform per trajectory.
        - batch_size (int): Size of mini-batches for updates.
        """
        values = self.value_net(observations).squeeze(-1)
        advantages = self.compute_advantages(rewards, values, dones)
        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Combine data into a single dataset for batching
        dataset = {
            "observations": observations,
            "actions": actions,
            "log_probs_old": log_probs_old,
            "returns": returns,
            "advantages": advantages,
        }

        # Perform multiple epochs of optimization
        for _ in range(epochs):
            for batch_idx in self._get_batches(len(observations), batch_size):
                # Mini-batch sampling
                batch = {k: (v[batch_idx] if isinstance(v, torch.Tensor) else [vi[batch_idx] for vi in v]) for k, v in
                         dataset.items()}

                # Recompute log probabilities and entropy
                new_distributions = self.policy_net(batch["observations"])
                new_log_probs = [
                    dist.log_prob(batch["actions"][i]) for i, dist in enumerate(new_distributions)
                ]
                entropy = torch.stack([dist.entropy() for dist in new_distributions]).mean()

                # Compute policy loss
                log_probs_new = torch.stack(new_log_probs).sum(dim=0)  # Detach graph here
                log_probs_old = torch.stack(batch["log_probs_old"]).sum(dim=0).detach()  # Detach old log-probs
                ratios = torch.exp(log_probs_new - log_probs_old)

                surr1 = ratios * batch["advantages"]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                values = self.value_net(batch["observations"]).squeeze(-1)
                value_loss = ((batch["returns"] - values) ** 2).mean()

                # Compute total loss
                loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

                # Perform optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def _get_batches(self, data_size, batch_size):
        """
        Generate mini-batch indices for splitting data.

        Args:
        - data_size (int): Total size of the data.
        - batch_size (int): Size of each mini-batch.

        Returns:
        - Generator: A generator yielding batch indices.
        """
        indices = torch.randperm(data_size)
        for start in range(0, data_size, batch_size):
            yield indices[start:start + batch_size]

# Example usage
if __name__ == "__main__":
    observation_dim = 8
    action_dims = [3, 5, 4]
    ppo = PPO(observation_dim, action_dims)

    # Example inputs
    observations = torch.randn(10, observation_dim)
    actions = [torch.randint(0, dim, (10,)) for dim in action_dims]
    log_probs_old = [torch.randn(10) for _ in action_dims]
    rewards = torch.randn(10)
    dones = torch.randint(0, 2, (10,))

    # PPO update
    # print weights
    prev_weights = ppo.policy_net.shared_layers[0].weight
    ppo.update(observations, actions, log_probs_old, rewards, dones)
    new_weights = ppo.policy_net.shared_layers[0].weight
    print(torch.allclose(prev_weights, new_weights))
