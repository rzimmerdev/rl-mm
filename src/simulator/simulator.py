from src.framework import Environment, Agent


class Simulator:

    def __init__(self, agent: Agent, environment: Environment):
        self.agent = agent
        self.environment = environment

    def run(self, duration: float = None):
        if duration is None:
            raise NotImplementedError("The simulator must be run for a specified duration")

        while not self.environment.end:
            action = self.agent.act(self.environment.state)
            next_state, reward, done = self.environment.step(action)
            self.agent.update(next_state, reward, done)

        return self.environment.state
