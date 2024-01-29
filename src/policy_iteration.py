import numpy as np


class State:
    def __init__(self, coord, is_terminal=False):
        self.coord = coord
        self.is_terminal = is_terminal

    @staticmethod
    def action_to_coord(action):
        # possible actions: 0=up, 1=right, 2=down, 3=left
        if action == 0:
            return np.array([-1, 0])
        elif action == 1:
            return np.array([0, 1])
        elif action == 2:
            return np.array([1, 0])
        elif action == 3:
            return np.array([0, -1])
        else:
            raise ValueError('Invalid action')


# Dynamic Programming Agent
class Agent:
    def __init__(self, gamma):
        self.gamma = gamma

        self.actions = [0, 1, 2, 3]
        # Grid world
        self.states = [State(coord=np.array([i, j])) for i in range(4) for j in range(4)]
        self.q = np.zeros((len(self.states), len(self.actions)))
        self.pi = np.zeros((len(self.states), len(self.actions)))

    def print_q(self):
        print(np.round(self.q, 2))

    def print_pi(self):
        # Print policy
        # possible actions: 0=up, 1=right, 2=down, 3=left
        # argmax over actions and print arrow

        pi_arrows = np.chararray((4, 4), unicode=True)

        for i in range(4):
            for j in range(4):
                if self.states[i * 4 + j].is_terminal:
                    pi_arrows[i, j] = 'T'
                else:
                    pi_arrows[i, j] = '↑→↓←'[np.argmax(self.pi[i * 4 + j])]

        print(pi_arrows)


class PolicyIterationAgent(Agent):
    def __init__(self, gamma):
        super().__init__(gamma)
        self.reward_terminal = 1

    def policy_evaluation(self, v, theta = 1e-4):
        delta = 0

        while delta > theta:
            v_old = v.copy()

            for idx, s in enumerate(self.states):
                if s.is_terminal:
                    v[idx] = self.reward_terminal
                else:
                    v[idx] = sum([
                        self.pi[s.coord][action] * (
                                0 + self.gamma * v_old[(s.coord + State.action_to_coord(action)).dot(np.array([4, 1]))])
                        for action in self.actions
                    ])

            delta = np.max(np.abs(v - v_old))

    def policy_improvement(self, v):
        stable = True

        for idx, s in enumerate(self.states):
            old_action = np.argmax(self.pi[idx])

            self.pi[idx] = np.argmax([
                sum([
                    self.pi[idx][action] * (
                            0 + self.gamma * v[(s.coord + State.action_to_coord(action)).dot(np.array([4, 1]))])
                    for action in self.actions
                ])
            ])

            if old_action != np.argmax(self.pi[idx]):
                stable = False

        return stable

    def policy_iteration(self, theta = 1e-4):
        v = np.zeros(len(self.states))

        while True:
            self.policy_evaluation(v, theta)
            stable = self.policy_improvement(v)

            if stable:
                break

        return v


def main():
    agent = PolicyIterationAgent(0.9)
    v = agent.policy_iteration()
    print(v.reshape(4, 4))

    agent.print_pi()

    agent = PolicyIterationAgent(0.5)
    v = agent.policy_iteration()

    print(v.reshape(4, 4))
    agent.print_pi()


if __name__ == '__main__':
    main()