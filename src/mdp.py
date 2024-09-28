class Action:
    def __init__(self, direction):
        if direction not in self.possible_directions:
            raise ValueError('Invalid action')
        self.direction = direction

    @property
    def coord(self):
        if self.direction == 0:
            return np.array([-1, 0])
        elif self.direction == 1:
            return np.array([0, 1])
        elif self.direction == 2:
            return np.array([1, 0])
        elif self.direction == 3:
            return np.array([0, -1])
        else:
            raise ValueError('Invalid action')


class State:
    def __init__(self, coord, is_terminal=False):
        self.coord = coord
        self.is_terminal = is_terminal

    @staticmethod
    def cap(coord, a_min=0, a_max=None):
        if a_max is None:
            a_max = len(Action.possible_directions) - 1
        return np.clip(coord, a_min, a_max)

    def add_coord(self, other: Action):
        return self.coord + other.coord
