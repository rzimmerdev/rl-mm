import os

import matplotlib.pyplot as plt


class RLTrainer:
    def __init__(self, env, agent, num_envs=4, checkpoints="checkpoints", rollout_length=2048):
        # self.envs = gym.vector.make(env_name, num_envs=num_envs, asynchronous=True)
        self.env = env
        self.agent = agent
        self.rollout_length = rollout_length
        self.checkpoints = checkpoints

        # get latest version and create a new one
        latest = sorted(os.listdir(checkpoints), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.newest_path = f'{checkpoints}/checkpoint_{len(latest)}/'
        self.latest_path = f'{checkpoints}/checkpoint_{len(latest) - 1}/' if len(latest) > 0 else None

    def train(self, num_episodes=1000):
        reward_history = []
        loss_history = []

        try:
            for episode in range(num_episodes):
                trajectories = self.agent.collect_trajectories(self.env)
                episode_loss = self.agent.update(trajectories)
                episode_reward = sum(trajectories.rewards)
                reward_history.append(episode_reward)
                loss_history.append(episode_loss)

                if episode % 10 == 0:
                    print(f'Episode {episode}, Reward: {episode_reward}, Loss: {episode_loss}')
        except KeyboardInterrupt:
            pass

        if not os.path.exists(self.newest_path):
            os.makedirs(self.newest_path)

        # store loss, rewards and agent weights
        self.agent.save_weights(f"{self.newest_path}/model.pth")
        self.plot_loss(loss_history)
        self.plot_rewards(reward_history)

    def load(self):
        if not self.latest_path:
            raise FileNotFoundError('No weights found')
        self.agent.load_weights(f"{self.latest_path}/model.pth")
        print(f'Loaded weights from {self.latest_path}')
        return self.agent

    def plot_loss(self, loss_history):
        plt.figure(figsize=(12, 6))
        plt.plot(loss_history, label='Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('PPO Training Progress')
        plt.legend()
        plt.grid(True)

        plt.savefig(f'{self.newest_path}/loss.png')

    def plot_rewards(self, reward_history):
        plt.figure(figsize=(12, 6))
        # remove outliers
        reward_history = [r for r in reward_history if abs(r) < 5000]
        plt.plot(reward_history, label='Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('PPO Training Progress')
        plt.legend()
        plt.grid(True)

        plt.savefig(f'{self.newest_path}/rewards.png')
