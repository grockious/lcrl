import os
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import tqdm
import time
import imageio
from environments.slippery_grid import SlipperyGrid


def animate(mdp, executed_policy, dir_to_save):
    if isinstance(mdp, SlipperyGrid):
        cmap = colors.ListedColormap(['red', 'black', 'blue', 'cyan', 'yellow'])
        bounds = [-2.9, -1.9, -0.9, 0.1, 1.1, 2.1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        animation_dir = os.path.join(dir_to_save, 'animation')
        if not os.path.exists(animation_dir):
            os.mkdir(animation_dir)
        print('---------------------------------\n')
        print('Creating a gif for the trained policy:')
        for i in tqdm.tqdm(range(len(executed_policy))):
            if i == 0:
                plt.imshow(mdp.labels, interpolation='nearest', cmap=cmap, norm=norm)
                path_x, path_y = np.array(executed_policy[0]).T
                plt.scatter(path_y, path_x, c='red', edgecolors='darkred')
            else:
                plt.imshow(mdp.labels, interpolation='nearest', cmap=cmap, norm=norm)
                path_x, path_y = np.array(executed_policy[0:i]).T
                plt.scatter(path_y, path_x, c='lime', edgecolors='teal')
                path_x, path_y = np.array(executed_policy[i]).T
                plt.scatter(path_y, path_x, c='red', edgecolors='darkred')
            plt.title('This policy is synthesised by the trained agent')
            plt.savefig(os.path.join(animation_dir, 'image_file_' + str(i) + '.png'))
        images = []
        for file_name in os.listdir(animation_dir):
            if file_name.endswith('.png'):
                file_path = os.path.join(animation_dir, file_name)
                images.append(imageio.imread(file_path))
        imageio.mimsave(os.path.join(animation_dir, 'executed_policy.gif'), images, fps=55)
