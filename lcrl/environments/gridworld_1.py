from lcrl.environments.slippery_grid import SlipperyGrid
import numpy as np

# an example slippery grid "assets/layout1.png" or "https://i.imgur.com/CzSbaYi.png"
# dark blue = safe
# red = unsafe
# cyan = goal1
# yellow = goal2
# only the labelling function needs to be specified

# create a SlipperyGrid object
gridworld_1 = SlipperyGrid(initial_state=[0, 0])


# "state_label" function outputs the label of input state (input: state, output: string label)
def state_label(self, state):
    # defines the labelling layout
    labels = np.empty([gridworld_1.shape[0], gridworld_1.shape[1]], dtype=object)
    labels[0:40, 0:40] = 'safe'
    labels[25:33, 7:15] = 'unsafe'
    labels[7:15, 25:33] = 'unsafe'
    labels[15:25, 15:25] = 'goal1'
    labels[33:40, 0:7] = 'goal2'

    # returns the label associated with input state
    return labels[state[0], state[1]]


# now override the step function
SlipperyGrid.state_label = state_label.__get__(gridworld_1, SlipperyGrid)
