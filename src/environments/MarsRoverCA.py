import os
import random
import numpy as np
from matplotlib.image import imread


class MarsRover:
    """
    An MDP whose labels depend on a background image

    ...

    Attributes
    ----------
    image: string
        path to a layout file

    Methods
    -------
    reset()
        resets the MDP state, including ghosts and agent
    step(action)
        changes the state of the MDP upon executing an action, where the action set is {right,up,left,down,stay}
    state_label(state)
        outputs the label of input state
    """

    def __init__(
            self,
            image='marsrover_1.png',
            initial_state=None
    ):
        self.background = imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'layouts', image))
        self.width = self.background.shape[0]
        self.height = self.background.shape[1]
        self.labels = self.background.__array__()
        self.initial_state = initial_state
        if self.initial_state is None:
            self.initial_state = np.array([60, 100], dtype=np.float32)
        self.current_state = self.initial_state
        # range for the sine of action angle direction
        self.action_space = [1, -1]

    def reset(self):
        self.current_state = self.initial_state.copy()

    def step(self, action):
        # agent movement dynamics:
        # # stochasticity
        traversed_distance = 4 + random.random()
        noise = np.array([random.uniform(-0.1, 0.5), random.uniform(-0.1, 0.5)])
        next_state = self.current_state + noise + \
                     np.append(traversed_distance * np.sin(action[0] * np.pi),
                               traversed_distance * np.cos(action[0] * np.pi))

        # check for boundary violations
        if next_state[0] > self.width - 1:
            next_state[0] = self.width - 1
        if next_state[1] > self.height - 1:
            next_state[1] = self.height - 1
        if next_state[0] < 0:
            next_state[0] = 0
        if next_state[1] < 0:
            next_state[1] = 0

        # update current state
        self.current_state = next_state
        return next_state

    def state_label(self, state):
        # note: labels are inevitably discrete when reading an image file
        # thus in the following we look where the continuous state lies within
        # the image rgb matrix
        low_bound = [abs(state[i] - int(state[i])) for i in range(len(state))]
        high_bound = [1 - low_bound[i] for i in range(len(state))]
        state_rgb_indx = []
        for i in range(len(state)):
            if low_bound[i] <= high_bound[i]:
                # check for boundary
                if int(state[i]) > self.background.shape[i] - 1:
                    state_rgb_indx.append(self.background.shape[i] - 1)
                else:
                    state_rgb_indx.append(int(state[i]))
            else:
                # check for boundary
                if int(state[i]) + 1 > self.background.shape[i] - 1:
                    state_rgb_indx.append(self.background.shape[i] - 1)
                else:
                    state_rgb_indx.append(int(state[i]) + 1)

        if list(self.labels[state_rgb_indx[0], state_rgb_indx[1]]) == list(self.labels[0, 199]) or \
                list(self.labels[state_rgb_indx[0], state_rgb_indx[1]]) == list(self.labels[144, 199]):
            return 'unsafe'
        elif list(self.labels[state_rgb_indx[0], state_rgb_indx[1]]) == list(self.labels[45, 152]):
            return 'goal1'
        elif list(self.labels[state_rgb_indx[0], state_rgb_indx[1]]) == list(self.labels[62, 16]):
            return 'goal2'
        else:
            return 'safe'
