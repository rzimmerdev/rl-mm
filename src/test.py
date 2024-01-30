import numpy as np

def main():
    n = 4
    grid = np.ones((n * n))
    grid *= -1

    # Upper Right
    grid[n - 1] = 5

    # Lower Left
    grid[-n] = 5

    print(grid.reshape((n, n)))

    policy = np.random.uniform(size=(n * n, 4))

    def get_action(policy, state):
        return np.argmax(policy[state])

    def move(state, action):
        coord = np.unravel_index(state, (n, n))
        if action == 0:
            d = np.array([-1, 0])
        elif action == 1:
            d = np.array([0, 1])
        elif action == 2:
            d = np.array([1, 0])
        elif action == 3:
            d = np.array([0, -1])
        else:
            raise ValueError("Invalid action")

        new_coord = np.clip(coord + d, 0, n - 1)
        return np.ravel_multi_index(new_coord, (n, n))

    def get_reward(state, action):
        return grid[move(state, action)]

    def get_transition(state, action):
        new_state = move(state, action)
        reward = get_reward(state, action)
        return new_state, reward

    V = np.zeros(n * n)
    Q = np.zeros((n * n, 4))

    gamma = 0.9

    while True:
        # Policy evaluation
        while True:
            v = V.copy()
            for s in range(n * n):
                Q[s] = np.array([get_transition(s, a)[1] + gamma * V[get_transition(s, a)[0]] for a in range(4)])
                V[s] = np.max(Q[s])

            delta = np.max(np.abs(v - V))
            print(delta)
            if delta < 1e-4:
                break

        # Policy improvement
        policy_stable = True
        for s in range(n * n):
            old_action = np.argmax(policy[s])
            policy[s] = np.zeros(4)
            policy[s, np.argmax(Q[s])] = 1
            if old_action != np.argmax(policy[s]):
                policy_stable = False

        if policy_stable:
            break

    # pretty print policy to arrows using argmax
    arrows = np.array(['↑→↓←'[np.argmax(policy[s])] for s in range(n * n)])
    arrows = arrows.reshape((n, n))
    print(arrows)
    print(V.reshape((n, n)))

if __name__ == "__main__":
   main()
