from src.environments.SlipperyGrid import SlipperyGrid
import numpy as np

# an example slippery grid "assets/layout3.png" or "https://i.imgur.com/CzSbaYi.png"
# dark blue = safe
# red = unsafe
# cyan = goal1
# yellow = goal2
# only the labelling function needs to be specified

# create a SlipperyGrid object
gridworld_3 = SlipperyGrid(shape=[5, 5], initial_state=[2, 0], slip_probability=0.05)

# defines the labelling image
labels = np.empty([gridworld_3.shape[0], gridworld_3.shape[1]], dtype=object)
labels[0:5, 0:5] = 'safe'
labels[1:4, 2] = 'obstacle'
labels[0:2, 3:5] = 'goal1'
labels[3:5, 3:5] = 'goal2'

# override the labels
gridworld_3.labels = labels
