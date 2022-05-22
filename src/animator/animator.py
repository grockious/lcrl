import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import imageio
from src.environments.SlipperyGrid import SlipperyGrid


def animate(mdp, executed_policy, dir_to_save, labels_value, cmap, norm, patches):
    if isinstance(mdp, SlipperyGrid):
        animation_dir = os.path.join(dir_to_save, 'animation')
        if not os.path.exists(animation_dir):
            os.mkdir(animation_dir)
        print('---------------------------------\n')
        print('Creating a gif for the trained policy:')
        for i in tqdm.tqdm(range(len(executed_policy))):
            if i == 0:
                plt.imshow(labels_value, interpolation='nearest', cmap=cmap, norm=norm)
                path_x, path_y = np.array(executed_policy[0]).T
                plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.scatter(path_y, path_x, c='red', edgecolors='darkred')
                plt.annotate('s_0', (path_y, path_x), fontsize=15, xytext=(20, 20), textcoords="offset points",
                             va="center", ha="left",
                             bbox=dict(boxstyle="round", fc="w"),
                             arrowprops=dict(arrowstyle="->"))
            else:
                plt.imshow(labels_value, interpolation='nearest', cmap=cmap, norm=norm)
                path_x, path_y = np.array(executed_policy[0:i]).T
                inits_y, initis_x = path_y[0], path_x[0]
                plt.scatter(path_y, path_x, c='lime', edgecolors='teal')
                path_x, path_y = np.array(executed_policy[i]).T
                plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.scatter(path_y, path_x, c='red', edgecolors='darkred')
                plt.annotate('s_0', (inits_y, initis_x), fontsize=15, xytext=(20, 20), textcoords="offset points",
                             va="center", ha="left",
                             bbox=dict(boxstyle="round", fc="w"),
                             arrowprops=dict(arrowstyle="->"))
            plt.title('This policy is synthesised by the trained agent')
            plt.savefig(os.path.join(animation_dir, 'image_file_' + str(i) + '.png'))
        images = []
        for file_name in os.listdir(animation_dir):
            if file_name.endswith('.png'):
                file_path = os.path.join(animation_dir, file_name)
                images.append(imageio.imread(file_path))
        imageio.mimsave(os.path.join(animation_dir, 'executed_policy.gif'), images, fps=55)
    else:
        raise NotImplementedError('The animator does not support this environment yet.')
