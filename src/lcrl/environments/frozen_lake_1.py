from lcrl.environments.SlipperyGrid import SlipperyGrid
import numpy as np

# A modified version of OpenAI Gym FrozenLake
# only the labelling function needs to be specified

# create a SlipperyGrid object
FrozenLake = SlipperyGrid(shape=[12, 10],
                          initial_state=[0, 5],
                          slip_probability=0.1,
                          sink_states=[[4, 7], [5, 8]]
                          )

# define the labellings
labels = np.empty([FrozenLake.shape[0], FrozenLake.shape[1]], dtype=object)
labels[0:12, 0:10] = 'safe'
labels[3:5, 3:5] = 'unsafe'
labels[4:6, 7:9] = 'goal1'
labels[7:9, 7:9] = 'goal2'
labels[6:8, 3:5] = 'goal3'
labels[7:9, 0:2] = 'goal4'

# override the labels
FrozenLake.labels = labels

# FrozenLake doesn't have the action "stay"
FrozenLake.action_space = [
    "right",
    "up",
    "left",
    "down",
]
