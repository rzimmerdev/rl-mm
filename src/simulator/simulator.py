import numpy as np
from rich.live import Live
from rich.table import Table

from src.framework import Environment, Agent


class Simulator:

    def __init__(self, agent: Agent, environment: Environment):
        self.agent = agent
        self.environment = environment

    def run(self):
        rewards = []
        non_zero_rewards = [0]
        with Live(refresh_per_second=4) as live:
            self.environment.start()
            while not self.environment.end:
                action = self.agent.act(self.environment.state)
                reward, done = self.environment.step(action)
                rewards.append(reward)
                if reward != 0:
                    non_zero_rewards.append(reward)
                self.agent.update(self.environment.state, reward, done)

                current_time = self.environment.duration - self.environment.remaining

                table = Table()
                table.add_column("Time", width=10)
                table.add_column("Action", width=50)
                table.add_column("Reward", width=10)

                table.add_row(
                    f"{round(current_time, 2)} / {self.environment.duration}",
                    f"{action[:2]}\n{action[2:]}, spread: {round((action[0] - action[2]) / 2, 4)}"
                    if action is not None else "None",
                    f"{round(non_zero_rewards[-1], 4)},\navg: {round(np.mean(non_zero_rewards), 2)}"
                )
                live.update(table)

        discounted_return = 0
        y = 0.9

        # for k in range(self.environment.duration):
        #     discounted_return += y ** k * rewards <--- use numpy to generate discount vector for gamma

        # 1. make vector with only y, then use np.cumprod to get the discount vector
        # 2. multiply the discount vector with the rewards vector
        # 3. sum
        discount_vector = np.full(len(rewards), y)
        discounted_return = np.dot(discount_vector, rewards)

        return self.environment.state, discounted_return
