import numpy as np
import random
from operator import add


# TODO: Define an abstract class for the MDP
class SlipperyGrid:
    """
    Slippery grid-world as MDP

    ...

    Attributes
    ----------
    layout : str
        labels layout over the grid (currently available: 'layout_1' 'layout_2')
    initial_state : list
        1d list with two elements (default for 'layout_1' and 'layout_2' is [0, 39])
    slip_probability: float
        probability of slipping (default 0.15)

    Methods
    -------
    reset()
        resets the MDP state
    step(action)
        changes the state of the MDP upon executing an action
    state_label(state)
        outputs the label of input state according to the layout attribute
    """

    def __init__(
            self, layout=None,
            initial_state=None,
            slip_probability=0.15,
    ):
        self.layout = layout
        if self.layout is None:
            raise Exception("please select a layout for the grid-world")
        elif self.layout not in ['layout_1', 'layout_2']:
            raise Exception("available layouts are: 'layout_1' and 'layout_2'")
        elif self.layout in ['layout_1', 'layout_2']:
            self.size = 40

            # layouts labelling matrix
            self.assignment = {
                'safe': 0,
                'goal_1': 1,
                'goal_2': 2,
                'unsafe': -2,
                'obstacle': -1
            }
            if self.layout == 'layout_1':  # assets/layout1.png
                self.labels = np.zeros([40, 40])
                self.labels[25:35, 5:15] = self.assignment['unsafe']
                self.labels[5:15, 25:35] = self.assignment['unsafe']
                self.labels[15:25, 15:25] = self.assignment['goal_1']
                self.labels[35:40, 0:5] = self.assignment['goal_2']
            if self.layout == 'layout_2':  # assets/layout1.png
                self.labels = np.zeros([40, 40])
                self.labels[25:33, 0:40] = self.assignment['unsafe']
                self.labels[25:33, 17:23] = self.assignment['safe']
                self.labels[0:8, 0:8] = self.assignment['goal_1']
                self.labels[33:40, 0:40] = self.assignment['goal_2']
            if initial_state is None:
                self.initial_state = [0, 39]

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
        if self.size in next_state:
            next_state[next_state.index(self.size)] = self.size - 1
        elif -1 in next_state:
            next_state[next_state.index(-1)] = 0

        # check for obstacles
        if self.state_label(next_state) == self.assignment['obstacle']:
            next_state = self.current_state

        # update current state
        self.current_state = next_state
        return next_state

    def state_label(self, state):
        return self.labels[state[0], state[1]]
