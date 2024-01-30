from typing import Tuple, List

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
    def __init__(self, actions: List[Action], states: List[State]):
        self.states = states
        self.actions = actions
        self.n = int(np.sqrt(len(states)))

        # Make terminal states upper right and lower left
        self.states[self.n - 1].is_terminal = True
        self.states[-self.n].is_terminal = True

    @property
    def num_states(self):
        return len(self.states)

    @property
    def num_actions(self):
        return len(self.actions)

    def reset(self):
        return self.states[0], False

    def _move(self, state: State, action: Action) -> Tuple[State, bool]:
        coord = State.cap(state.coord + action.coord, a_max=self.n - 1)
        state = self.states[np.ravel_multi_index(coord, (self.n, self.n))]
        return state, state.is_terminal

    def move(self, s: int, a: int) -> Tuple[int, bool]:
        state = self.states[s]
        action = self.actions[a]
        state, done = self._move(state, action)
        return self.states.index(state), done

    def _p(self, state: State, action: Action, s_prime: State = None) -> float | List[float]:
        if s_prime is None:
            return [self._p(state, action, s_prime) for s_prime in self.states]
        return 1 if np.all(self._move(state, action)[0].coord == s_prime.coord) else 0


    def p(self, s: int, a: int, s_: int = None) -> float | List[float]:
        if s_ is None:
            return [self.p(s, a, s_) for s_ in range(self.num_states)]
        return 1 if np.all(self.move(s, a)[0] == s_) else 0

    def step(self, s: int, a: int) -> Tuple[int, float, bool]:
        p = self.p(s, a)

        s_ = np.random.choice(self.num_states, p=p)
        reward = self.reward(s_, s, a)
        done = self.states[s_].is_terminal

        return s_, reward, done

    def _reward(self, state_prime: State, _: State = None, __: Action = None, deterministic=True) -> float:
        # Rewards are deterministic
        if deterministic:
            return 5 if state_prime.is_terminal else -1
        else:
            return 5 if np.random.uniform() < 0.5 else -1

    def reward(self,  s_: int, s: int = None, a: int = None, deterministic=True) -> float:
        return self._reward(self.states[s_], self.states[s], self.actions[a], deterministic)


# Dynamic Programming Agent
class Agent:
    def __init__(self, environment: Environment, gamma=0.9):
        self.environment = environment
        self.gamma = gamma

        self.policy = None
        self.V = None
        self.Q = None

        self.set()

    def get_action(self, s) -> int:
        return int(np.argmax(self.policy[s]))

    def set(self):
        # probability to take action a given state s π(a|s)
        self.policy = np.random.uniform(size=(self.environment.num_states, self.environment.num_actions))
        self.policy /= self.environment.num_actions

        # Uppercase is the approximated value of a state given policy π using the Bellman equation (DP approach)
        self.V = np.zeros(self.environment.num_states)
        self.Q = np.zeros((self.environment.num_states, self.environment.num_actions))

    def print_arrows(self):
        # possible actions: 0=up, 1=right, 2=down, 3=left
        # argmax over actions and print arrow
        n = self.environment.n
        arrows = np.array(['↑→↓←'[np.argmax(self.policy[s])] for s in range(n * n)])
        print(np.reshape(arrows, (n, n)))

    def print_V(self):
        # Print V
        n = self.environment.n
        V = self.V.reshape((n, n))
        print(V)


class PolicyIterationAgent(Agent):

    def policy_evaluation(self, V, Q, theta=1e-4, max_iter=1e3):
        V = V.copy()
        Q = Q.copy()

        delta = 0
        for _ in range(int(max_iter)):
            v = V.copy()

            for s in range(self.environment.num_states):
                for a in range(self.environment.num_actions):
                    s_, reward, _ = self.environment.step(s, a)
                    Q[s, a] = reward + self.gamma * V[s_]

                V[s] = np.max(Q[s])

            delta = np.max(np.abs(v - V))
            print(delta)
            if delta < theta:
                break

        return V, Q, delta

    def policy_improvement(self, policy, Q):
        stable = True
        policy = policy.copy()

        for s in range(self.environment.num_states):
            old_action = np.argmax(policy[s])

            policy[s] = np.zeros(self.environment.num_actions)
            policy[s, np.argmax(Q[s])] = 1

            if old_action != np.argmax(policy[s]):
                stable = False

        return stable, policy

    def policy_iteration(self, theta=1e-4, max_iter=1e3):
        t = 0

        for _ in range(int(max_iter)):
            self.V, self.Q, delta = self.policy_evaluation(self.V, self.Q, theta, max_iter)
            stable, self.policy = self.policy_improvement(self.policy, self.Q)

            if t % 10 == 0:
                print(f'Iteration {t}')
                print(f'Delta: {delta}')
            t += 1

            if stable:
                break

        return self.policy


class Simulation:
    def __init__(self, agent: Agent, environment: Environment):
        self.agent = agent
        self.environment = environment

    def simulate(self, num_steps=1e3):
        state, done = self.environment.reset()
        s = self.environment.states.index(state)
        total_reward = 0
        time_step = 0

        steps = []

        while not done and time_step < num_steps:
            a = self.agent.get_action(s)
            steps.append((s, a))
            s_, reward, done = self.environment.step(s, a)
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
    actions = [Action(i) for i in range(4)]
    # Grid world
    states = [State(coord=np.array([i, j])) for i in range(n) for j in range(n)]

    environment = Environment(actions, states)
    agent = PolicyIterationAgent(environment)

    agent.policy_iteration()
    agent.print_arrows()
    agent.print_V()

    simulation = Simulation(agent, environment)
    total_reward, done = simulation.simulate()
    print(f'Total reward: {total_reward}')


if __name__ == '__main__':
    main()
