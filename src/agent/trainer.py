

class RLTrainer:
    def __init__(self, env, agent, num_envs=4, rollout_length=2048):
        # self.envs = gym.vector.make(env_name, num_envs=num_envs, asynchronous=True)
        self.env = env
        self.agent = agent
        self.rollout_length = rollout_length

    def train(self, num_episodes=1000):
        reward_history = []

        try:
            for episode in range(num_episodes):
                trajectories = self.agent.collect_trajectories(self.env)
                episode_loss = self.agent.update(trajectories)
                episode_reward = sum(trajectories.rewards)
                reward_history.append(episode_reward)

                if episode % 10 == 0:
                    print(f'Episode {episode}, Reward: {episode_reward}, Loss: {episode_loss}')
        except KeyboardInterrupt:
            pass

        return reward_history