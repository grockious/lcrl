from lcrl.environments.SlipperyGrid import SlipperyGrid
import numpy as np

# an example slippery grid
# only the labelling function needs to be specified

# create a SlipperyGrid object
slp_med = SlipperyGrid(shape=[20, 20], initial_state=[2, 0], slip_probability=0.05)

# define the labellings
labels = np.empty([slp_med.shape[0], slp_med.shape[1]], dtype=object)
labels[0:20, 0:20] = 'safe'
labels[12:16, 15:19] = 'goal1'
labels[15:19, 15:19] = 'goal2'
labels[9:13, 9:13] = 'goal3'
labels[0:4, 15:19] = 'goal4'

# override the labels
slp_med.labels = labels