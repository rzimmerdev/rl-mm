# agent/environment.py
# import dxlib as dx


# TODO: Add trading functionalities to dxlib
class Environment:
    """
    Environment class for simulating a reinforcement learning trading agent.

    Parameters
    ----------
    data : dx.Frame
        Data to be used for the environment.
    config : dict
        Configuration dictionary for the environment.
    """
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.reset()

        # self.action_space = dx.Discrete(2)

        self.t = 0
        self.done = False
        self.position = 0
        self.cost = 0
        self.trades = 0
        self.state = None

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.t = 0
        self.done = False
        self.position = 0
        self.cost = 0
        self.trades = 0
        self._update_state()

        return self.state

    def _update_state(self):
        """
        Update the state of the environment.
        """
        # self.state = dx.Frame({
        #     'prices': self.data['prices'].iloc[self.t],
        #     'position': self.position,
        #     'cost': self.cost,
        #     'trades': self.trades
        # })
        pass

    def step(self, action):
        """
        Take a step in the environment.

        Parameters
        ----------
        action : int
            Action to take in the environment.

        Returns
        -------
        state : dx.Frame
            State of the environment after taking the step.
        reward : float
            Reward for taking the step.
        done : bool
            Whether the episode has ended.
        info : dict
            Additional information about the step.
        """
        if self.done:
            return self.state, 0, self.done, {}

        reward = 0

        if action == 1:
            self.position = 1
            self.cost = self.data['prices'].iloc[self.t]

        return self.state, reward, self.done, {}
