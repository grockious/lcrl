from src.environments.slippery_grid import SlipperyGrid
import numpy as np

# an example slippery grid
# only the labelling function needs to be specified

# create a SlipperyGrid object
slp_sml = SlipperyGrid(shape=[12, 10], initial_state=[2, 0], slip_probability=0.05)

# define the labellings
labels = np.empty([slp_sml.shape[0], slp_sml.shape[1]], dtype=object)
labels[0:12, 0:10] = 'safe'
labels[5][8] = labels[5][9] = labels[6][8] = labels[6][9] = 'goal1'
labels[8][8] = labels[8][9] = labels[9][8] = labels[9][9] = 'goal2'
labels[7][4] = labels[7][5] = labels[8][4] = labels[8][5] = 'goal3'
labels[8][0] = labels[8][1] = labels[9][0] = labels[9][1] = 'goal4'

# override the labels
slp_sml.labels = labels
