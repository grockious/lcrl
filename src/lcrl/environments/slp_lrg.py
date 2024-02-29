from src.environments.SlipperyGrid import SlipperyGrid
import numpy as np

# an example slippery grid
# only the labelling function needs to be specified

# create a SlipperyGrid object
slp_lrg = SlipperyGrid(shape=[40, 40], initial_state=[2, 0], slip_probability=0.05)

# define the labellings
labels = np.empty([slp_lrg.shape[0], slp_lrg.shape[1]], dtype=object)
labels[0:40, 0:40] = 'safe'
labels[20:28, 31:39] = 'goal1'
labels[31:39, 31:39] = 'goal2'
labels[20:28, 20:28] = 'goal3'
labels[0:8, 31:39] = 'goal4'

# override the labels
slp_lrg.labels = labels