from lcrl.environments.SlipperyGrid import SlipperyGrid
import numpy as np

# an example slippery grid "assets/layout1.png" or "https://i.imgur.com/CzSbaYi.png"
# dark blue = safe
# red = unsafe
# cyan = goal1
# yellow = goal2
# only the labelling function needs to be specified

# create a SlipperyGrid object
gridworld_2 = SlipperyGrid(initial_state=[0, 39])

# defines the labelling image
labels = np.empty([gridworld_2.shape[0], gridworld_2.shape[1]], dtype=object)
labels[0:40, 0:40] = 'safe'
labels[25:33, 0:40] = 'unsafe'
labels[25:33, 12:28] = 'safe'
labels[0:8, 0:8] = 'goal1'
labels[33:40, 0:40] = 'goal2'

# override the labels
gridworld_2.labels = labels

