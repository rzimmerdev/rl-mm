import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Layer initialization helper.
    """
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_units=(64, 64), hidden_activation=nn.ReLU(), output_activation=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_units)):
            if i == 0:
                self.layers.append(layer_init(nn.Linear(in_dim, hidden_units[i])))
            else:
                self.layers.append(layer_init(nn.Linear(hidden_units[i - 1], hidden_units[i])))
            self.layers.append(hidden_activation)
        self.layers.append(layer_init(nn.Linear(hidden_units[-1], out_dim)))
        if output_activation:
            self.layers.append(output_activation)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            layer_init(nn.Linear(in_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, out_dim), std=0.01),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state):
        action_probs = self(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, dist, dist.log_prob(action)


class MultiHeadPolicyNetwork(nn.Module):
    def __init__(self, in_dim, out_dims):
        super(MultiHeadPolicyNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU()
        )
        self.output_layers = nn.ModuleList([
            nn.Sequential(layer_init(nn.Linear(64, dim)), nn.Softmax(dim=-1)) for dim in out_dims
        ])

    def forward(self, observations):
        shared_features = self.shared_layers(observations)
        probs = [layer(shared_features) for layer in self.output_layers]
        return probs

    def act(self, state):
        probs = self(state)
        dist = [Categorical(prob) for prob in probs]
        action = [d.sample() for d in dist]
        return action, dist, [d.log_prob(a) for d, a in zip(dist, action)]


class ValueNetwork(nn.Module):
    def __init__(self, in_dim):
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            layer_init(nn.Linear(in_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1)
        )

    def forward(self, x):
        return self.layers(x)


def shape_sanity():
    batch_size = 5
    in_dim = 10
    out_dims = [2, 3, 4]
    model = MultiHeadPolicyNetwork(in_dim, out_dims)
    x = torch.randn(batch_size, in_dim)
    probs = model(x)
    assert len(probs) == len(out_dims)
    for i, prob in enumerate(probs):
        assert prob.shape == (batch_size, out_dims[i])
    actions, dist, log_probs = model.act(x)
    assert len(actions) == len(out_dims)
    assert len(log_probs) == len(out_dims)
    for i, action in enumerate(actions):
        assert action.shape == (batch_size,)


def backward_sanity():
    batch_size = 5
    in_dim = 10
    out_dim = 2
    model = PolicyNetwork(in_dim, out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(batch_size, in_dim)
    y = torch.randint(0, out_dim, (batch_size,))
    weights = model.layers[2].weight.clone().detach()
    for _ in range(10):
        optimizer.zero_grad()
        action, dist, log_prob = model.act(x)
        loss = -dist.log_prob(y).mean()
        loss.backward()
        optimizer.step()
    new_weights = model.layers[2].weight

    assert not torch.allclose(weights, new_weights)


if __name__ == "__main__":
    shape_sanity()
    backward_sanity()
    print("All tests passed")
