class Hawkes:
    def __init__(self, mean=.1, alpha=.2, beta=.1):
        """
        Hawkes process with exponential kernel

        Args:
            mean: base rate of the process, the larger the mean, the more events
        """
        self.mean = mean
        self.alpha = alpha
        self.beta = beta
        self.events = np.array([])

    def reset(self):
        self.events = np.array([])

    def register_event(self, t):
        self.events = np.append(self.events, t)

    def phi(self, x):
        return self.alpha * self.beta * np.exp(-self.beta * x)

    def intensity(self, t):
        """
        Returns the intensity of the process at time t, the intensity is the rate of events for the current time.
        """
        return self.mean + np.sum(self.phi(t - (self.events[self.events < t])))

    def __call__(self, t):
        """
        Sample the time until the next event
        """
        try:
            return np.random.exponential(1 / self.intensity(t))
        except ZeroDivisionError:
            return np.inf

    def cdf(self, t):
        return 1 - np.exp(-self.intensity(t))