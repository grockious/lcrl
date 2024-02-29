import os
import random
import numpy as np
from lcrl.environments.SlipperyGrid import SlipperyGrid
from lcrl.automata.ldba import LDBA
from lcrl.core.lcrl_core import LCRL
from lcrl.animator.animator import animate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
from datetime import datetime


def train(
        MDP,
        LDBA,
        algorithm='ql',
        episode_num=2500,
        iteration_num_max=4000,
        discount_factor=0.95,
        learning_rate=0.9,
        nfq_replay_buffer_size=50,
        ddpg_replay_buffer_size=100000,
        decaying_learning_rate=False,
        epsilon=0.1,
        save_dir='./results',
        test=True,
        average_window=-1,
):
    learning_task = LCRL(MDP, LDBA, discount_factor, learning_rate, decaying_learning_rate, epsilon)

    if algorithm == 'ql':
        learning_task.train_ql(episode_num, iteration_num_max)
        import dill
        from lcrl.environments.MarsRoverDA import MarsRover
    elif algorithm == 'nfq':
        learning_task.train_nfq(episode_num, iteration_num_max, nfq_replay_buffer_size)
        import dill
        from lcrl.environments.MarsRoverDA import MarsRover
    elif algorithm == 'ddpg':
        learning_task.train_ddpg(episode_num, iteration_num_max, ddpg_replay_buffer_size)
        import dill
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from lcrl.environments.MarsRoverCA import MarsRover
    else:
        raise NotImplementedError('New learning algorithms will be added to LCRL soon. The selected algorithm is not '
                                  'implemented yet.')

    if average_window == -1:
        average_window = int(0.03 * episode_num)

    plt.plot(learning_task.q_at_initial_state, c="royalblue")
    plt.xlabel('Episode Number')
    plt.ylabel('Value Function at The Initial State')
    plt.grid(True)
    if average_window > 0:
        avg = np.convolve(learning_task.q_at_initial_state, np.ones((average_window,)) / average_window, mode='valid')
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

    if test:
        print('testing...')
        number_of_tests = 100
        learning_task.successes_in_test = 0
        for tt in range(number_of_tests):
            MDP.reset()
            LDBA.reset()
            # check if MDP current_state is a list or ndarray:
            if type(MDP.current_state) == np.ndarray:
                ndarray = True
                test_path = [MDP.current_state.tolist()]
            else:
                ndarray = False
                test_path = [MDP.current_state]
            iteration_num = 0
            while LDBA.accepting_frontier_set and iteration_num < iteration_num_max \
                    and LDBA.automaton_state != -1:
                iteration_num += 1
                if ndarray:
                    if algorithm == "nfq":
                        current_state = MDP.current_state.tolist() + [LDBA.automaton_state]
                    if algorithm == "ddpg":
                        prev_state = MDP.current_state
                        current_state = MDP.current_state.tolist() + [LDBA.automaton_state]
                else:
                    current_state = MDP.current_state + [LDBA.automaton_state]

                if learning_task.epsilon_transitions_exists:
                    product_MDP_action_space = learning_task.action_space_augmentation()
                else:
                    product_MDP_action_space = MDP.action_space

                if not algorithm == "ddpg":
                    Qs = []
                    if (not ndarray) and (str(current_state) in learning_task.Q.keys()):
                        for action_index in range(len(product_MDP_action_space)):
                            Qs.append(learning_task.Q[str(current_state)][product_MDP_action_space[action_index]])
                    elif ndarray:
                        for action_index in range(len(product_MDP_action_space)):
                            Qs.append(learning_task.Q[current_state[-1]].predict(
                                [MDP.current_state.tolist() + [action_index]]))
                    else:
                        Qs.append(0)
                    maxQ_action_index = random.choice(np.where(Qs == np.max(Qs))[0])
                    maxQ_action = product_MDP_action_space[maxQ_action_index]
                    # check if an epsilon-transition is taken
                    if learning_task.epsilon_transitions_exists and \
                            maxQ_action_index > len(learning_task.MDP.action_space) - 1:
                        epsilon_transition_taken = True
                    else:
                        epsilon_transition_taken = False
                    if epsilon_transition_taken:
                        next_MDP_state = learning_task.MDP.current_state if not ndarray else learning_task.MDP.current_state.tolist()
                        next_automaton_state = learning_task.LDBA.step(maxQ_action)
                    else:
                        next_MDP_state = learning_task.MDP.step(maxQ_action)
                        next_automaton_state = learning_task.LDBA.step(learning_task.MDP.state_label(next_MDP_state))
                        if ndarray:
                            next_MDP_state = next_MDP_state.tolist()
                else:
                    # action space bounds
                    lower_bound = -1.0
                    upper_bound = 1.0
                    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                    sampled_actions = tf.squeeze(learning_task.Q[current_state[-1]](tf_prev_state))
                    sampled_actions = sampled_actions.numpy()
                    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
                    action = [np.squeeze(legal_action)]
                    if learning_task.epsilon_transitions_exists and \
                            LDBA.automaton_state in LDBA.epsilon_transitions.keys() and \
                            random.random() > 0.5:
                        epsilon_action = random.choice(product_MDP_action_space[2:])
                        action = [np.squeeze(
                            int(epsilon_action[-1]) + learning_task.upper_bound
                        )]
                        epsilon_transition_taken = True
                    else:
                        epsilon_transition_taken = False
                    # product MDP modification (for more details refer to https://bit.ly/LCRL_CDC_2019)
                    if epsilon_transition_taken:
                        next_automaton_state = LDBA.step(epsilon_action)
                        next_MDP_state = prev_state
                    else:
                        state = MDP.step(action)
                        next_MDP_state = state.tolist()
                        next_automaton_state = LDBA.step(MDP.state_label(next_MDP_state))

                    # product MDP: synchronise the automaton with MDP
                    prev_state = state.copy()

                test_path.append(next_MDP_state)
                if not epsilon_transition_taken:
                    LDBA.accepting_frontier_function(next_automaton_state)

                if not LDBA.accepting_frontier_set:
                    learning_task.successes_in_test += 1

        print('success rate in testing: ' + str(100 * learning_task.successes_in_test / number_of_tests) + '%')

    if isinstance(MDP, SlipperyGrid) and test:
        # plt.plot(learning_task.path_length, c='royalblue')
        # plt.xlabel('Episode Number')
        # plt.ylabel('Agent Traversed Distance from The Initial State')
        # plt.grid(True)
        # if average_window > 0:
        #     avg = np.convolve(learning_task.path_length, np.ones((average_window,)) / average_window, mode='valid')
        #     plt.plot(avg, c='darkblue')
        # plt.savefig(os.path.join(results_sub_path, 'traversed distance in the grid.png'))
        # plt.show()

        distinct_labels = np.unique(learning_task.MDP.labels)
        labels_dic = {}
        label_indx = 0
        bounds = [-0.9]
        cmap = plt.get_cmap('gist_rainbow')
        for label in distinct_labels:
            labels_dic[label] = label_indx
            bounds.append(bounds[-1] + 1)
            label_indx += 1
        color_map = cmap(np.linspace(0, 1, len(distinct_labels)))
        cmap = colors.ListedColormap(color_map)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        labels_value = np.zeros([learning_task.MDP.shape[0], learning_task.MDP.shape[1]])
        for i in range(learning_task.MDP.shape[0]):
            for j in range(learning_task.MDP.shape[1]):
                labels_value[i][j] = labels_dic[learning_task.MDP.state_label([i, j])]
        patches = [mpatches.Patch(color=color_map[i], label=list(distinct_labels)[i]) for i in
                   range(len(distinct_labels))]
        plt.imshow(labels_value, interpolation='nearest', cmap=cmap, norm=norm)
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        path_x, path_y = np.array(test_path).T
        plt.scatter(path_y, path_x, c='lime', edgecolors='teal')
        plt.scatter(path_y[0], path_x[0], c='red', edgecolors='black')
        plt.annotate('s_0', (path_y[0], path_x[0]), fontsize=15, xytext=(20, 20), textcoords="offset points",
                     va="center", ha="left",
                     bbox=dict(boxstyle="round", fc="w"),
                     arrowprops=dict(arrowstyle="->"))
        plt.title('This policy is synthesised by the trained agent')
        plt.savefig(
            os.path.join(results_sub_path, 'tested_policy.png'), bbox_inches="tight")
        plt.show()
        is_gif = input(
            'Would you like to create a gif for the the control policy? '
            'If so, type in "y", otherwise, type in "n". ')
        if is_gif == 'y' or is_gif == 'Y':
            animate(learning_task.MDP, test_path, results_sub_path, labels_value, cmap, norm, patches)
        print('\n---------------------------------\n')
        print('The results have been saved here:\n')
        print(results_sub_path)
        return learning_task

    if isinstance(MDP, MarsRover) and test:
        plt.imshow(MDP.background)
        path_x, path_y = np.array(test_path).T
        plt.scatter(path_y, path_x, c='lime', edgecolors='teal')
        plt.scatter(path_y[0], path_x[0], c='red', edgecolors='black')
        plt.annotate('s_0', (path_y[0], path_x[0]), fontsize=15, xytext=(20, 20), textcoords="offset points",
                     va="center", ha="left",
                     bbox=dict(boxstyle="round", fc="w"),
                     arrowprops=dict(arrowstyle="->"))
        plt.title('This policy is synthesised by the trained agent')
        plt.savefig(
            os.path.join(results_sub_path, 'tested_policy.png'), bbox_inches="tight")
        plt.show()
        is_gif = input(
            'Would you like to create a gif for the the control policy? '
            'If so, type in "y", otherwise, type in "n". ')
        if is_gif == 'y' or is_gif == 'Y':
            animate(learning_task.MDP, test_path, results_sub_path, labels_value, cmap, norm, patches)
        print('\n---------------------------------\n')
        print('The results have been saved here:\n')
        print(results_sub_path)
        return learning_task

    if algorithm == 'ql':
        with open(os.path.join(results_sub_path, 'learned_model.pkl'), 'wb') as learning_file:
            dill.dump(learning_task, learning_file)
        if test:
            with open(os.path.join(results_sub_path, 'test_results.pkl'), 'wb') as test_file:
                dill.dump(test_path, test_file)
        print('In order to load the learning results use the following command in Python console:')
        print('import dill')
        print("learned_model = dill.load(open('" + os.path.join(results_sub_path, 'learned_model.pkl') + "', 'rb'))")
        if test:
            print("tested_trace = dill.load(open('" + os.path.join(results_sub_path, 'test_results.pkl') + "', 'rb'))")
        print('\n---------------------------------\n')
        if learning_task.early_interruption == 0:
            print("Training finished successfully!")
        else:
            print("Training results have been saved successfully! [Note: training was interrupted by user]")
        return learning_task
    # TODO: change the save method and add nfq & ddpg
    return learning_task
