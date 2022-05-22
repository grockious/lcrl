from src.environments.SlipperyGrid import SlipperyGrid
import numpy as np

# A modified version of OpenAI Gym FrozenLake
# only the labelling function needs to be specified

# create a SlipperyGrid object
FrozenLake = SlipperyGrid(shape=[40, 40],
                          initial_state=[0, 20],
                          slip_probability=0.1
                          )

# define the labellings
labels = np.empty([FrozenLake.shape[0], FrozenLake.shape[1]], dtype=object)
labels[0:40, 0:40] = 'safe'
labels[4:8, 9:13] = 'unsafe'
labels[20:28, 31:39] = 'goal1'
labels[31:39, 31:39] = 'goal2'
labels[20:28, 20:28] = 'goal3'
labels[0:4, 15:19] = 'goal4'

# override the labels
FrozenLake.labels = labels

# FrozenLake doesn't have the action "stay"
FrozenLake.action_space = [
    "right",
    "up",
    "left",
    "down",
]