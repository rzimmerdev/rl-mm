from tqdm import tqdm

from src.framework import Environment, Agent


class Simulator:

    def __init__(self, agent: Agent, environment: Environment):
        self.agent = agent
        self.environment = environment

    def run(self):
        # use tqdm with environment.duration and environment.remaining
        rewards = 0
        with tqdm(total=self.environment.duration) as pbar:
            prev_time = self.environment.duration - self.environment.remaining
            self.environment.start()
            while not self.environment.end:
                action = self.agent.act(self.environment.state)
                reward, done = self.environment.step(action)
                rewards += reward
                self.agent.update(self.environment.state, reward, done)

                current_time = self.environment.duration - self.environment.remaining
                pbar.update(current_time - prev_time)
                prev_time = current_time

        discounted_return = 0
        y = 0.9
        for k in range(self.environment.duration):
            discounted_return += y ** k * rewards

        return self.environment.state, discounted_return
