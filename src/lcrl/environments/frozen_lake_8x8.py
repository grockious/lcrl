from lcrl.environments.SlipperyGrid import SlipperyGrid
import numpy as np

# OpenAI Gym FrozenLake 8x8
# only the labelling function needs to be specified

# create a SlipperyGrid object
FrozenLake = SlipperyGrid(shape=[8, 8], initial_state=[0, 0], slip_probability=0.05, sink_states=[[7, 7]])

# defines the labelling image
labels = np.empty([FrozenLake.shape[0], FrozenLake.shape[1]], dtype=object)
labels[0:8, 0:8] = 'safe'
labels[2][3] = labels[3][5] = labels[4][3] = labels[5][1] = 'unsafe'
labels[5][2] = labels[5][6] = labels[6][1] = 'unsafe'
labels[6][4] = labels[6][6] = labels[7][3] = 'unsafe'
labels[0][7] = 'goal1'
labels[7][7] = 'goal2'

# override the labels
FrozenLake.labels = labels


FrozenLake.action_space = [
    "right",
    "up",
    "left",
    "down",
]

