import unittest

from env.market_env import MarketEnv


class TestMarketEnv(unittest.TestCase):
    def test_state(self):
        env = MarketEnv()

        state = env._get_state()
        shape = state.shape
        levels = env.n_levels
        assert shape == (5 + 4 * levels,)
        assert (state >= 0).all()

    def test_act(self):
        env = MarketEnv()

        action = env.action_space.sample()
        next_state, reward, done, _, info = env.step(action)
        assert env.observation_space.contains(next_state)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        print(reward)

if __name__ == '__main__':
    unittest.main()
