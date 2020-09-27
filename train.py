import os
import random
import numpy as np
import argparse
from environments.slippery_grid import SlipperyGrid
from automata.ldba import LDBA
from scripts.lcrl import LCRL
from animator import animate
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    required_parse = parser.add_argument_group('required arguments')
    required_parse.add_argument('-env', '--env', metavar='ENVIRONMENT', required=True,
                                help="choose an environment from the currently available env set {SlipperyGrid}, or alternatively define a custom one within '/environments' folder")
    required_parse.add_argument('-layout', '--layout', metavar='LAYOUT', required=True,
                                help="choose a label map from the currently available layouts {SlipperyGrid: 'layout_1' or 'layout_2'}, or alternatively define a custom one inside the current environment file")
    required_parse.add_argument('-property', '--property', metavar='TEMPORAL_PROPERTY', required=True,
                                help="choose a temporal properties from the currently available properties for each env {SlipperyGrid: 'g1-then-g2'(goal_1 then goal_2), 'g1-or-g2'(goal_1 or goal_2)}, or alternatively define a custom one inside automata/ldba.py")
    parser.add_argument('-save_dir', '--save_dir', metavar='SAVE DIRECTORY', default='./results', type=str,
                        help='Directory to save the results')
    parser.add_argument('-alg', '--algorithm', metavar='ALGORITHM', default='ql', type=str,
                        help='Learning algorithm is chosen automatically based on the input env but can be overridden')
    parser.add_argument('-df', '--discount_factor', metavar='DISCOUNT_FACTOR', default=0.99, type=float,
                        help='Discount factor')
    parser.add_argument('-lr', '--learning_rate', metavar='LEARNING_RATE', default=0.9, type=float,
                        help='Learning rate')
    parser.add_argument('-ep', '--epsilon', metavar='EPSILON', default=0.0, type=float,
                        help='Hyper-parameter for epsilon-greedy exploration')
    parser.add_argument('-episode', '--episode_number', metavar='EPISODE_NUMBER', default=1000, type=int,
                        help='Number of learning episodes')
    parser.add_argument('-iteration', '--iteration', metavar='ITERATION_NUM', default=4000, type=int,
                        help='The threshold of number of iterations in each learning episode')
    parser.add_argument('-test', '--test', metavar='TEST_FOR_FINAL_RESULT', default=True, type=bool,
                        help='True for testing and rendering the final policy')
    parser.add_argument('-avgw', '--avgw', metavar='AVERAGER_WINDOW', default=0, type=int,
                        help='True for testing and rendering the final policy')
    hyper_parameters = parser.parse_args()
    return hyper_parameters


def main():
    hyper_parameters = parse_args()
    env = hyper_parameters.env
    layout = hyper_parameters.layout
    temporal_property = hyper_parameters.property
    save_dir = hyper_parameters.save_dir
    algorithm = hyper_parameters.algorithm
    gamma = hyper_parameters.discount_factor
    alpha = hyper_parameters.learning_rate
    epsilon = hyper_parameters.epsilon
    episode_num = hyper_parameters.episode_number
    iteration_num_threshold = hyper_parameters.iteration
    test = hyper_parameters.test
    average_window = hyper_parameters.avgw

    automaton = LDBA(automaton_state=0, temporal_property=temporal_property)

    # # # import custom env here # # #
    if env == 'SlipperyGrid':
        algorithm = 'ql'
        MDP = SlipperyGrid(layout)
    else:
        raise NotImplementedError('Other environments need to be defined and imported to train.py')
    # # # # # # # # # # # # # # # # #

    learning = LCRL(MDP, automaton, gamma, alpha, epsilon)

    if algorithm == 'ql':
        learning.train_ql(episode_num, iteration_num_threshold)
        import dill
    else:
        raise NotImplementedError

    plt.plot(learning.path_length, c='royalblue')
    plt.xlabel('Episode Number')
    plt.ylabel('Agent Traversed Distance')
    plt.grid(True)
    if average_window > 0:
        avg = np.convolve(learning.path_length, np.ones((average_window,)) / average_window, mode='valid')
        plt.plot(avg, c='darkblue')

    # saving the results
    results_path = os.path.join(os.getcwd(), save_dir[2:])
    dt_string = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    results_sub_path = os.path.join(os.getcwd(), save_dir[2:], dt_string)
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    os.mkdir(results_sub_path)
    plt.savefig(os.path.join(results_sub_path, 'convergence_' + env + '_' + layout + '_' + temporal_property + '.png'))

    plt.show()
    if isinstance(MDP, SlipperyGrid) and test:
        learning.MDP.reset()
        learning.LDBA.reset()
        test_path = [learning.MDP.current_state]
        epsilon_transition_taken = False
        iteration_num = 0
        while learning.LDBA.accepting_frontier_set and iteration_num < iteration_num_threshold \
                and learning.LDBA.automaton_state != -1:
            iteration_num += 1
            current_state = learning.MDP.current_state + [learning.LDBA.automaton_state]
            if learning.epsilon_transitions_exists:
                product_MDP_action_space = learning.action_space_augmentation()
            else:
                product_MDP_action_space = MDP.action_space
            Qs = []
            if str(current_state) in learning.Q.keys():
                for action_index in range(len(product_MDP_action_space)):
                    Qs.append(learning.Q[str(current_state)][product_MDP_action_space[action_index]])
            else:
                Qs.append(0)
            maxQ_action_index = random.choice(np.where(Qs == np.max(Qs))[0])
            maxQ_action = product_MDP_action_space[maxQ_action_index]
            # check if an epsilon-transition is taken
            if learning.epsilon_transitions_exists and \
                    maxQ_action_index > len(learning.MDP.action_space) - 1:
                epsilon_transition_taken = True
            if epsilon_transition_taken:
                next_MDP_state = learning.MDP.current_state
                next_automaton_state = learning.LDBA.step(maxQ_action)
            else:
                next_MDP_state = learning.MDP.step(maxQ_action)
            test_path.append(next_MDP_state)
            next_automaton_state = learning.LDBA.step(learning.MDP.state_label(next_MDP_state))
            learning.LDBA.accepting_frontier_function(next_automaton_state)
            next_state = next_MDP_state + [next_automaton_state]
            current_state = next_state
        cmap = colors.ListedColormap(['red', 'black', 'blue', 'cyan', 'yellow'])
        bounds = [-2.9, -1.9, -0.9, 0.1, 1.1, 2.1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(learning.MDP.labels, interpolation='nearest', cmap=cmap, norm=norm)
        path_x, path_y = np.array(test_path).T
        plt.scatter(path_y, path_x, c='lime', edgecolors='teal')
        plt.title('This policy is synthesised by the trained agent')
        plt.savefig(
            os.path.join(results_sub_path, 'tested_policy_' + env + '_' + layout + '_' + temporal_property + '.png'))
        plt.show()
        animate(learning.MDP, test_path, results_sub_path)
        print('---------------------------------\n')
        print('The results have been saved here:\n')
        print(results_sub_path)

    if algorithm == 'ql':
        with open(os.path.join(results_sub_path, 'learning.pkl'), 'wb') as learning_file:
            dill.dump(learning, learning_file)
        with open(os.path.join(results_sub_path, 'test.pkl'), 'wb') as test_file:
            dill.dump(test_path, test_file)
        print('to load the learning results use the following command in Python console:')
        print("learned_model = dill.load(open('" + os.path.join(results_sub_path, 'learning.pkl') + "', 'rb'))")
        print("tested_trace = dill.load(open('" + os.path.join(results_sub_path, 'test.pkl') + "', 'rb'))")
        print('---------------------------------\n')
        print("Training Finished Successfully!")
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
