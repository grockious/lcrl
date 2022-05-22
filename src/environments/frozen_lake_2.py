from src.environments.SlipperyGrid import SlipperyGrid
import numpy as np

# A modified version of OpenAI Gym FrozenLake
# only the labelling function needs to be specified
sinks = []
for i in range(12, 16):
    for j in range(15, 19):
        sinks.append([i, j])

# create a SlipperyGrid object
FrozenLake = SlipperyGrid(shape=[20, 20],
                          initial_state=[0, 10],
                          slip_probability=0.1,
                          sink_states=sinks
                          )

# define the labellings
labels = np.empty([FrozenLake.shape[0], FrozenLake.shape[1]], dtype=object)
labels[0:20, 0:20] = 'safe'
labels[4:8, 9:13] = 'unsafe'
labels[12:16, 15:19] = 'goal1'
labels[15:19, 15:19] = 'goal2'
labels[9:13, 9:13] = 'goal3'
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
