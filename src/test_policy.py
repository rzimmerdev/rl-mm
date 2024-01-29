import unittest

import numpy as np

from policy_iteration import PolicyIterationAgent, Environment, Action, State


class TestEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        self.actions = {i: Action(i) for i in range(4)}
        self.states = {(i, j): State(coord=np.array([i, j])) for i in range(4) for j in range(4)}

        self.environment = Environment(self.actions, self.states)

    def test_state_idx(self):
        state = self.states[(2, 1)]
        idx = self.environment.state_idx(state)
        self.assertEqual(idx, 9)

    def test_idx_state(self):
        state_idx = 9
        state = self.environment.idx_state(state_idx)
        self.assertEqual(state.coord.tolist(), [2, 1])

    def test_reset(self):
        initial_state = self.environment.reset()
        self.assertEqual(initial_state.coord.tolist(), [0, 0])

    def test_p(self):
        old_state = self.states[(1, 1)]
        action = self.actions[0]
        new_state = self.states[(0, 1)]

        prob = self.environment._p(old_state, action, new_state)
        self.assertEqual(prob, 1.0)

    def test_t(self):
        old_state = self.states[(1, 1)]
        pi = np.ones((16, 4)) / 4
        new_state = self.states[(0, 1)]

        prob = self.environment._t(old_state, pi, new_state)
        self.assertEqual(prob, 0.25)

    def test_step(self):
        old_state = self.states[(1, 1)]
        action = self.actions[0]

        new_state, done = self.environment._step(old_state, action)
        new_state = self.environment.idx_state(new_state)
        self.assertEqual(new_state.coord.tolist(), [0, 1])
        self.assertFalse(done)

    def test_reward(self):
        old_state = self.states[(1, 1)]
        action = self.actions[0]

        reward = self.environment._reward(old_state, action)
        self.assertEqual(reward, -1)


class TestPolicyIterationAgent(unittest.TestCase):
    def setUp(self):
        self.actions = {i: Action(i) for i in range(4)}
        self.states = {(i, j): State(coord=np.array([i, j])) for i in range(4) for j in range(4)}

        self.environment = Environment(self.actions, self.states)
        self.agent = PolicyIterationAgent(self.environment)


    def test_policy_evaluation(self):
        V, delta = self.agent.policy_evaluation(theta=1e-4, max_iter=10, scaling=4)

        # Add your assertions here based on expected values

    def test_policy_improvement(self):
        stable = self.agent.policy_improvement(scaling=4)

        # Add your assertions here based on expected values

    def test_policy_iteration(self):
        optimal_policy = self.agent.policy_iteration(theta=1e-4, max_iter=10)

        # Add your assertions here based on expected values


if __name__ == '__main__':
    unittest.main()
