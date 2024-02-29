from src.environments.SlipperyGrid import SlipperyGrid
import numpy as np

# only the labelling function needs to be specified

# create a SlipperyGrid object
robot_surv = SlipperyGrid(shape=[5, 5], initial_state=[1, 0], slip_probability=0.05)

# define the labellings
labels = np.empty([robot_surv.shape[0], robot_surv.shape[1]], dtype=object)
labels[0:5, 0:5] = 'safe'
labels[0, 0] = labels[1, 2] = labels[1, 3] = labels[4, 2] = 'obstacle'
labels[4, 0] = 'goal1'
labels[0, 4] = 'goal2'

# override the labels
robot_surv.labels = labels
