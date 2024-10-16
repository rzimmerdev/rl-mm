from tqdm import tqdm

from src.framework import Environment, Agent


class Simulator:

    def __init__(self, agent: Agent, environment: Environment):
        self.agent = agent
        self.environment = environment

    def run(self):
        # use tqdm with environment.duration and environment.remaining
        with tqdm(total=self.environment.duration) as pbar:
            prev_time = self.environment.duration - self.environment.remaining
            while not self.environment.end:
                action = self.agent.act(self.environment.state)
                next_state, reward, done = self.environment.step(action)
                self.agent.update(next_state, reward, done)

                current_time = self.environment.duration - self.environment.remaining
                pbar.update(current_time - prev_time)
                prev_time = current_time

        return self.environment.state
