import os
import csv
from torch.utils.tensorboard import SummaryWriter


class RLTrainer:
    def __init__(self, env, agent, num_envs=4, checkpoints="checkpoints", rollout_length=2048):
        self.env = env
        self.agent = agent
        self.rollout_length = rollout_length
        self.checkpoints = checkpoints

        # Ensure checkpoint directory exists
        if not os.path.exists(checkpoints):
            os.makedirs(checkpoints)

        latest = sorted(os.listdir(checkpoints), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.newest_path = f'{checkpoints}/checkpoint_{len(latest)}/'
        self.latest_path = f'{checkpoints}/checkpoint_{len(latest) - 1}/' if len(latest) > 0 else None
        self.latest_path = f'{checkpoints}/checkpoint_{len(latest) - 1}/' if len(latest) > 0 else None

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir="runs/RLTrainer")

    def train(self, num_episodes=1000):
        reward_history = []
        loss_history = []

        if not os.path.exists(self.newest_path):
            os.makedirs(self.newest_path)

        snapshots = {col: [] for col in self.env.snapshot_columns}
        snapshots["episode"] = []

        try:
            for episode in range(num_episodes):
                trajectories = self.agent.collect_trajectories(self.env, self.rollout_length)
                episode_loss = self.agent.update(trajectories)
                episode_reward = sum(trajectories.rewards)

                reward_history.append(episode_reward)
                loss_history.append(episode_loss)

                # Log metrics to TensorBoard
                self.writer.add_scalar("Reward", episode_reward, episode)
                self.writer.add_scalar("Loss", episode_loss, episode)

                snapshot = self.env.snapshot()
                snapshot["episode"] = episode
                for col in snapshots.keys():
                    snapshots[col].append(snapshot[col])

                if episode % 10 == 0:
                    print(f'Episode {episode}, Reward: {episode_reward}, Loss: {episode_loss}')
        except KeyboardInterrupt:
            pass

        self.writer.close()  # Close writer when training is done

        # Save snapshots and metrics
        with open(f'{self.newest_path}/snapshots.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=snapshots.keys())
            writer.writeheader()
            for i in range(len(snapshots['financial_return'])):
                writer.writerow({col: snapshots[col][i] for col in snapshots.keys()})

        self.agent.save_weights(f"{self.newest_path}/model.pth")

        with open(f'{self.newest_path}/metrics.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['episode', 'reward', 'loss'])
            writer.writeheader()
            for i in range(len(reward_history)):
                writer.writerow({'episode': i, 'reward': reward_history[i], 'loss': loss_history[i]})

    def load(self):
        if not self.latest_path:
            raise FileNotFoundError('No weights found')
        self.agent.load_weights(f"{self.latest_path}/model.pth")
        print(f'Loaded weights from {self.latest_path}')
        return self.agent
