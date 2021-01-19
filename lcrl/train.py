import os
import random
import numpy as np
from lcrl.environments.slippery_grid import SlipperyGrid
from lcrl.automata.ldba import LDBA
from lcrl.core.lcrl_core import LCRL
from lcrl.animator.animator import animate
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import datetime


def train(
        MDP,
        LDBA,
        algorithm='ql',
        episode_num=1000,
        iteration_num_max=4000,
        save_dir='./results',
        discount_factor=0.9,
        learning_rate=0.9,
        epsilon=0.1,
        test=True,
        average_window=-1,
):

    learning_task = LCRL(MDP, LDBA, discount_factor, learning_rate, epsilon)

    if algorithm == 'ql':
        learning_task.train_ql(episode_num, iteration_num_max)
        import dill
    else:
        raise NotImplementedError('New learning algorithms will be added to lcrl_core.py soon.')

    if average_window == -1:
        average_window = int(0.03 * episode_num)

    plt.plot(learning_task.path_length, c='royalblue')
    plt.xlabel('Episode Number')
    plt.ylabel('Agent Traversed Distance')
    plt.grid(True)
    if average_window > 0:
        avg = np.convolve(learning_task.path_length, np.ones((average_window,)) / average_window, mode='valid')
        plt.plot(avg, c='darkblue')

    # saving the results
    results_path = os.path.join(os.getcwd(), save_dir[2:])
    dt_string = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    results_sub_path = os.path.join(os.getcwd(), save_dir[2:], dt_string)
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    os.mkdir(results_sub_path)
    plt.savefig(os.path.join(results_sub_path, 'convergence.png'))

    plt.show()
    if isinstance(MDP, SlipperyGrid) and test:
        learning_task.MDP.reset()
        learning_task.LDBA.reset()
        test_path = [learning_task.MDP.current_state]
        epsilon_transition_taken = False
        iteration_num = 0
        while learning_task.LDBA.accepting_frontier_set and iteration_num < iteration_num_max \
                and learning_task.LDBA.automaton_state != -1:
            iteration_num += 1
            current_state = learning_task.MDP.current_state + [learning_task.LDBA.automaton_state]
            if learning_task.epsilon_transitions_exists:
                product_MDP_action_space = learning_task.action_space_augmentation()
            else:
                product_MDP_action_space = MDP.action_space
            Qs = []
            if str(current_state) in learning_task.Q.keys():
                for action_index in range(len(product_MDP_action_space)):
                    Qs.append(learning_task.Q[str(current_state)][product_MDP_action_space[action_index]])
            else:
                Qs.append(0)
            maxQ_action_index = random.choice(np.where(Qs == np.max(Qs))[0])
            maxQ_action = product_MDP_action_space[maxQ_action_index]
            # check if an epsilon-transition is taken
            if learning_task.epsilon_transitions_exists and \
                    maxQ_action_index > len(learning_task.MDP.action_space) - 1:
                epsilon_transition_taken = True
            if epsilon_transition_taken:
                next_MDP_state = learning_task.MDP.current_state
                next_automaton_state = learning_task.LDBA.step(maxQ_action)
            else:
                next_MDP_state = learning_task.MDP.step(maxQ_action)
            test_path.append(next_MDP_state)
            next_automaton_state = learning_task.LDBA.step(learning_task.MDP.state_label(next_MDP_state))
            learning_task.LDBA.accepting_frontier_function(next_automaton_state)
            next_state = next_MDP_state + [next_automaton_state]
            current_state = next_state
        cmap = colors.ListedColormap(['red', 'black', 'blue', 'cyan', 'yellow'])
        bounds = [-2.9, -1.9, -0.9, 0.1, 1.1, 2.1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        labels_dic = {
                'safe': 0,
                'goal1': 1,
                'goal2': 2,
                'unsafe': -2,
                'obstacle': -1
            }
        labels_value = np.zeros([learning_task.MDP.shape[0], learning_task.MDP.shape[1]])
        for i in range(learning_task.MDP.shape[0]):
            for j in range(learning_task.MDP.shape[1]):
                labels_value[i][j] = labels_dic[learning_task.MDP.state_label([i, j])]

        plt.imshow(labels_value, interpolation='nearest', cmap=cmap, norm=norm)
        path_x, path_y = np.array(test_path).T
        plt.scatter(path_y, path_x, c='lime', edgecolors='teal')
        plt.title('This policy is synthesised by the trained agent')
        plt.savefig(
            os.path.join(results_sub_path, 'tested_policy.png'))
        plt.show()
        is_gif = input(
            'Would you like to create a gif for the the control policy? '
            'If so, type in "y", otherwise, type in "n". ')
        if is_gif == 'y' or is_gif == 'Y':
            animate(learning_task.MDP, test_path, results_sub_path, labels_value)
        print('\n---------------------------------\n')
        print('The results have been saved here:\n')
        print(results_sub_path)
    else:
        raise NotImplementedError

    if algorithm == 'ql':
        with open(os.path.join(results_sub_path, 'learning.pkl'), 'wb') as learning_file:
            dill.dump(learning_task, learning_file)
        with open(os.path.join(results_sub_path, 'test.pkl'), 'wb') as test_file:
            dill.dump(test_path, test_file)
        print('In order to load the learning results use the following command in Python console:')
        print("learned_model = dill.load(open('" + os.path.join(results_sub_path, 'learning.pkl') + "', 'rb'))")
        print("tested_trace = dill.load(open('" + os.path.join(results_sub_path, 'test.pkl') + "', 'rb'))")
        print('\n---------------------------------\n')
        if learning_task.early_interruption == 0:
            print("Training finished successfully!")
        else:
            print("Training results have been saved successfully! [Note: training was interrupted by user]")
    else:
        raise NotImplementedError
