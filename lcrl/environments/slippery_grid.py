import random
from operator import add


class SlipperyGrid:
    """
    Slippery grid-world modelled as an MDP

    ...

    Attributes
    ----------
    shape: list
        1d list with two elements: 1st element is the num of row cells and the 2nd is the num of column cells (default [40, 40])
    initial_state : list
        1d list with two elements (default [0, 39])
    slip_probability: float
        probability of slipping (default 0.15)

    Methods
    -------
    reset()
        resets the MDP state
    step(action)
        changes the state of the MDP upon executing an action, where the action set is {right,up,left,down,stay}
    state_label(state)
        outputs the label of input state
    """

    def __init__(
            self,
            shape=None,
            initial_state=None,
            slip_probability=0.15,
    ):
        if shape is None:
            self.shape = [40, 40]
        else:
            self.shape = shape
        if initial_state is None:
            self.initial_state = [0, 39]
        else:
            self.initial_state = initial_state

        self.current_state = self.initial_state
        self.slip_probability = slip_probability

        # directional actions
        self.action_space = [
            "right",
            "up",
            "left",
            "down",
            "stay"
        ]

    def reset(self):
        self.current_state = self.initial_state

    def step(self, action):
        # slipperiness
        if random.random() < self.slip_probability:
            action = random.choice(self.action_space)

        # grid movement dynamics:
        if action == 'right':
            next_state = list(map(add, self.current_state, [0, 1]))
        elif action == 'up':
            next_state = list(map(add, self.current_state, [-1, 0]))
        elif action == 'left':
            next_state = list(map(add, self.current_state, [0, -1]))
        elif action == 'down':
            next_state = list(map(add, self.current_state, [1, 0]))
        elif action == 'stay':
            next_state = self.current_state

        # check for boundary violations
        if next_state[0] == self.shape[0]:
            next_state[0] = self.shape[0] - 1
        if next_state[1] == self.shape[1]:
            next_state[1] = self.shape[1] - 1
        if -1 in next_state:
            next_state[next_state.index(-1)] = 0

        # update current state
        self.current_state = next_state
        return next_state

    def state_label(self, state):
        pass
