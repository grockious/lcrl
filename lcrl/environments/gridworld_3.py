from lcrl.environments.slippery_grid import SlipperyGrid
import numpy as np
import random

# an example slippery grid "assets/layout3.png" or "https://i.imgur.com/CzSbaYi.png"
# dark blue = safe
# red = unsafe
# cyan = goal1
# yellow = goal2
# only the labelling function needs to be specified

# create a SlipperyGrid object
gridworld_3 = SlipperyGrid(shape=[5, 5], initial_state=[2, 0], slip_probability=0.05)


# "state_label" function outputs the label of input state (input: state, output: string label)
def state_label(self, state):
    # defines the labelling layout
    labels = np.empty([gridworld_3.shape[0], gridworld_3.shape[1]], dtype=object)
    labels[0:5, 0:5] = 'safe'
    labels[1:4, 2] = 'obstacle'
    labels[0:2, 3:5] = 'goal1'
    labels[3:5, 3:5] = 'goal2'

    # returns the label associated with input state
    return labels[state[0], state[1]]


# now override the step function
SlipperyGrid.state_label = state_label.__get__(gridworld_3, SlipperyGrid)
