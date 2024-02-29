from lcrl.environments.SlipperyGrid import SlipperyGrid
import numpy as np

# create a SlipperyGrid object
minecraft = SlipperyGrid(shape=[10, 10], initial_state=[9, 2], slip_probability=0)

# define the labellings
labels = np.empty([minecraft.shape[0], minecraft.shape[1]], dtype=object)
labels[0:10, 0:10] = 'safe'
labels[0:3, 5] = 'obstacle'
labels[2, 7:10] = 'obstacle'
labels[0][0] = labels[4][5] = labels[8][1] = labels[8][7] = 'grass'
labels[2][2] = labels[7][3] = labels[5][7] = labels[9][9] = 'wood'
labels[0][3] = labels[4][0] = labels[6][8] = labels[9][4] = 'iron'
labels[6][1] = labels[6][5] = labels[4][9] = 'work_bench'
labels[2][4] = labels[9][0] = labels[7][7] = 'tool_shed'
labels[0][7] = 'gold'

# override the labels
minecraft.labels = labels
