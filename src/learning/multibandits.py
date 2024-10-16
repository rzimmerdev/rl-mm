# Author: Rafael Zimmer
import numpy as np
import pandas as pd
from plotly import express as px


class MultiBandit:
    def __init__(self, k=10, reward_std=1):
        self.k = k
        self.reward_std = reward_std
        self.mean_values = np.random.normal(loc=0, scale=1, size=k)

    def reward(self, action):
        if not 0 <= action < self.k:
            raise ValueError('Invalid action')
        return np.random.normal(loc=self.mean_values[action], scale=self.reward_std)


class Agent:
    def __init__(self, eps, k=10):
        self.eps = eps
        self.k = k

    def policy(self, Q):
        if np.random.rand() < self.eps:
            return np.random.randint(low=0, high=self.k)
        else:
            return np.argmax(Q)


class TestBed:
    def __init__(self, agent, bandits):
        self.agent = agent
        self.bandits = bandits

    def experiment(self, num_steps):
        Q = np.zeros(self.agent.k)
        N = np.zeros(self.agent.k)

        rewards = np.zeros(num_steps)
        actions = np.zeros(num_steps)
        best_action = np.argmax(self.bandits.mean_values)

        for t in range(num_steps):
            action = self.agent.policy(Q)
            reward = self.bandits.reward(action)

            actions[t] = ((actions[t - 1] if t > 0 else 0) + int(action == best_action))
            rewards[t] = reward

            N[action] += 1
            Q[action] += 1 / N[action] * (reward - Q[action])

        actions /= np.arange(1, num_steps + 1)

        return Q, rewards, actions

    def mean(self, num_experiments=1e3, num_steps=1e3):
        num_steps = int(num_steps)
        num_experiments = int(num_experiments)

        rewards = np.zeros((num_experiments, num_steps))
        actions = np.zeros((num_experiments, num_steps))

        for i in range(num_experiments):
            _, experiment_rewards, experiment_actions = self.experiment(num_steps)
            rewards[i] = experiment_rewards
            actions[i] = experiment_actions

        return np.mean(rewards, axis=0), np.mean(actions, axis=0)


bed1 = TestBed(Agent(0), MultiBandit())
bed2 = TestBed(Agent(1e-1), MultiBandit())
bed3 = TestBed(Agent(1e-2), MultiBandit())

x = np.arange(0, int(1e3))

results = np.array([bed1.mean(), bed2.mean(), bed3.mean()])
y_actions = results[:, 0]
y_rewards = results[:, 1]

df_actions = pd.DataFrame(y_actions.T, columns=["eps=0", "eps=1e-1", "1e-2"])
df_rewards = pd.DataFrame(y_rewards.T, columns=["eps=0", "eps=1e-1", "1e-2"])

fig = px.line(df_rewards)
fig.show()
fig = px.line(df_actions)
fig.show()
