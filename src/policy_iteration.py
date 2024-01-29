from typing import Dict, Tuple

import numpy as np


class Action:
    possible_directions = [0, 1, 2, 3]

    def __init__(self, direction):
        if direction not in self.possible_directions:
            raise ValueError('Invalid action')
        self.direction = direction

    @property
    def coord(self):
        if self.direction == 0:
            return np.array([-1, 0])
        elif self.direction == 1:
            return np.array([0, 1])
        elif self.direction == 2:
            return np.array([1, 0])
        elif self.direction == 3:
            return np.array([0, -1])
        else:
            raise ValueError('Invalid action')


class State:
    def __init__(self, coord, is_terminal=False):
        self.coord = coord
        self.is_terminal = is_terminal

    @staticmethod
    def cap(coord, a_min=0, a_max=None):
        if a_max is None:
            a_max = len(Action.possible_directions) - 1
        return np.clip(coord, a_min, a_max)

    def add_coord(self, other: Action):
        return self.coord + other.coord


class Environment:
    def __init__(self, actions: Dict[int, Action], states: Dict[tuple, State]):
        self.states = states
        self.actions = actions
        self.n = int(np.sqrt(len(states)))

        # Make terminal states upper right and lower left
        self.states[(self.n - 1, 0)].is_terminal = True
        self.states[(0, self.n - 1)].is_terminal = True

    @property
    def num_states(self):
        return len(self.states)

    @property
    def num_actions(self):
        return len(self.actions)

    def state_idx(self, state: State) -> int:
        # States as a 2D array, return idx in 1D array
        return state.coord.dot(np.array([self.n, 1]))

    def idx_state(self, idx: int) -> State:
        # 1D idx to 2D array
        return self.states[(idx // self.n, idx % self.n)]

    def reset(self):
        return self.states[(0, 0)]

    def _p(self, old_state: State, action: Action, new_state: State = None) -> float | Dict[State, float]:
        # Transition probabilities are deterministic
        if new_state is not None:
            if np.all(State.cap(old_state.coord + action.coord) == new_state.coord):
                return 1
            return 0
        else:
            return {
                new_state: self._p(old_state, action, new_state)
                for new_state in self.states.values()
            }

    def p(self, old_state: int, action: int, new_state: int = None) -> float | Dict[int, float]:
        # Given action, probability of transitioning from old_state to new_state
        old_state = self.idx_state(old_state)
        action = self.actions[action]

        if new_state is not None:
            new_state = self.idx_state(new_state)

            return self._p(old_state, action, new_state)
        else:
            return {
                self.state_idx(new_state): prob
                for new_state, prob in self._p(old_state, action).items()
            }

    def _t(self, old_state: State, pi: np.ndarray, new_state: State) -> float:
        # Probability of transitioning from old_state to new_state given policy pi
        # p(s'|s,π(s))
        # Using the Total probability law
        # pi = [state, action] -> probability
        idx_old_state = self.state_idx(old_state)
        return float(np.sum([
            pi[idx_old_state][idx_action] * self._p(old_state, action, new_state)
            for idx_action, action in self.actions.items()
        ]))

    def t(self, old_state: int, pi: np.ndarray, new_state: int) -> float:
        # Given policy pi, probability of transitioning from old_state to new_state
        # Sums over all actions and their probabilities of being taken * probability of that action actually leading to
        return float(np.sum([
            pi[old_state][action] * self.p(old_state, action, new_state)
            for action in self.actions
        ]))

    def _step(self, state: State, action: Action) -> Tuple[int, bool]:
        p: Dict[State, float] = self._p(state, action)

        # Use probability to sample next state
        # p = {State: prob}
        # must transform state to coord
        states = list(self.states.values())
        probs = {states.index(s): p[s] for s in states}
        idx_new_state = np.random.choice(list(probs.keys()), p=list(probs.values()))

        is_terminal = self.idx_state(idx_new_state).is_terminal

        return idx_new_state, is_terminal

    def step(self, state: int, action: int) -> Tuple[int, bool]:
        state = self.idx_state(state)
        action = self.actions[action]

        idx_new_state, is_terminal = self._step(state, action)

        return idx_new_state, is_terminal

    def _reward(self, old_state: State, action: Action) -> float:
        # Rewards are deterministic
        new_state, done = self._step(old_state, action)

        if done:
            return 5
        else:
            return -1

    def reward(self, old_state: int, action: int) -> float:
        old_state = self.idx_state(old_state)
        action = self.actions[action]

        return self._reward(old_state, action)


# Dynamic Programming Agent
class Agent:
    def __init__(self, environment: Environment, gamma=0.3):
        self.environment = environment
        self.gamma = gamma

        self.policy = None
        self.V_memo = None
        self.Q_memo = None

        self.set()

    def set(self):
        # probability to take action a given state s π(a|s)
        self.policy = np.ones((self.environment.num_states, self.environment.num_actions))
        self.policy /= self.environment.num_actions

        # Uppercase is the approximated value of a state given policy π using the Bellman equation (DP approach)
        self.V_memo = np.zeros(self.environment.num_states)

    def print_policy(self):
        # Print policy
        # possible actions: 0=up, 1=right, 2=down, 3=left
        # argmax over actions and print arrow

        arrows = np.chararray((self.environment.n, self.environment.n), unicode=True)

        for coord, state in self.environment.states.items():
            s = self.environment.state_idx(state)
            arrows[coord] = '↑→↓←'[np.argmax(self.policy[s])]

        print(arrows)

    def print_V(self):
        # Print V
        print(self.V_memo.reshape((self.environment.n, self.environment.n)))


class PolicyIterationAgent(Agent):
    def policy_evaluation(self, theta=1e-4, max_iter=10, scaling=4):
        delta = 0
        for _ in range(max_iter):

            v = self.V_memo.copy()
            self.V_memo = np.array([
                np.sum([
                    self.environment.t(s, self.policy, s_) * (
                            self.environment.reward(s, a) + self.gamma * self.V_memo[s_]
                    )
                    for a in range(self.environment.num_actions)
                    for s_ in range(self.environment.num_states)
                ])
                for s in range(self.environment.num_states)
            ])

            self.V_memo = np.round(self.V_memo, 2) / scaling

            delta = np.max(np.abs(v - self.V_memo))
            if delta < theta:
                break

        return self.V_memo, delta

    def policy_improvement(self, scaling=4):
        stable = True

        for s in range(self.environment.num_states):
            old_action = np.argmax(self.policy[s])

            Q = np.array([
                np.sum([
                    self.environment.t(s, self.policy, s_) * (
                            self.environment.reward(s, a) + self.gamma * self.V_memo[s_]
                    )
                    for s_ in range(self.environment.num_states)
                ])
                for a in range(self.environment.num_actions)
            ])

            Q = np.round(Q, 2) / scaling

            best_action = np.argmax(Q)
            self.policy[s] = np.eye(self.environment.num_actions)[best_action]

            if old_action != best_action:
                stable = False

        return stable

    def policy_iteration(self, theta=1e-4, max_iter=1e1):
        stable = False
        t = 0

        while not stable and t < max_iter:
            _, delta = self.policy_evaluation(theta)
            stable = self.policy_improvement()
            if t % 10 == 0:
                print(f'Iteration {t}')
                print(f'Delta: {delta}')
            t += 1

        return self.policy


class Simulation:
    def __init__(self, agent: Agent, environment: Environment):
        self.agent = agent
        self.environment = environment

    def simulate(self, num_steps=1e3):
        state = self.environment.reset()
        s = self.environment.state_idx(state)
        done = state.is_terminal
        total_reward = 0
        time_step = 0

        steps = []

        while not done:
            a = np.argmax(self.agent.policy[s])
            steps.append((s, a))
            s_, done = self.environment.step(s, a)
            reward = self.environment.reward(s, a)
            total_reward += reward
            s = s_

            time_step += 1
            if time_step > num_steps:
                break

        print("Trajectory:")
        # map to arrows
        print([
            '↑→↓←'[a]
            for _, a in steps
        ])

        return total_reward, done


def main():
    n = 4
    actions = {i: Action(i) for i in range(4)}
    # Grid world
    states = {(i, j): State(coord=np.array([i, j])) for i in range(n) for j in range(n)}

    environment = Environment(actions, states)
    agent = PolicyIterationAgent(environment)

    agent.policy_iteration()
    agent.print_policy()
    agent.print_V()

    # Simulation
    simulation = Simulation(agent, environment)
    total_reward, done = simulation.simulate()
    print(f'Total reward: {total_reward}')


if __name__ == '__main__':
    main()
