from src.environments.slippery_grid import SlipperyGrid
import numpy as np

# OpenAI Gym FrozenLake 4x4
# only the labelling function needs to be specified

# create a SlipperyGrid object
FrozenLake = SlipperyGrid(shape=[4, 4], initial_state=[0, 0], slip_probability=0.05, sink_states=[[3, 3]])

# define the labellings
labels = np.empty([FrozenLake.shape[0], FrozenLake.shape[1]], dtype=object)
labels[0:4, 0:4] = 'safe'
labels[1, 1] = labels[1, 3] = labels[2, 3] = labels[3, 0] = 'unsafe'
labels[0, 3] = 'goal1'
labels[3, 3] = 'goal2'

# override the labels
FrozenLake.labels = labels

# FrozenLake doesn't have the action "stay"
FrozenLake.action_space = [
    "right",
    "up",
    "left",
    "down",
]
