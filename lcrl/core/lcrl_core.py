import numpy as np
import random


class LCRL:
    """
    LCRL

    ...

    Attributes
    ----------
    MDP : an object with the following properties
        (1) a "step(action)" function describing the dynamics
        (2) a "reset()" function that resets the state to an initial state
        (3) a "state_label(state)" function that maps states to labels
        (4) current state of the MDP is "current_state"
        (5) action space is "action_space" and all actions are enabled in each state
    LDBA : an object of ./lcrl/automata/ldba.py
        a limit-deterministic Büchi automaton
    discount_factor: float
        discount factor (default 0.9)
    learning_rate: float
        learning rate (default 0.9)
    epsilon: float
        tuning parameter for the epsilon-greedy exploration scheme (default 0.1)

    Methods
    -------
    train_ql(number_of_episodes, iteration_threshold, Q_initial_value)
        employs Episodic Q-learning to synthesise an optimal policy over the product MDP
        for more details refer to https://bit.ly/LCRL_CDC_2019
    reward(automaton_state)
        shapes the reward function according to the automaton accepting frontier set
        for more details refer to https://bit.ly/LCRL_CDC_2019
    """

    def __init__(
            self, MDP=None,
            LDBA=None,
            discount_factor=0.9,
            learning_rate=0.9,
            epsilon=0.15
    ):
        if MDP is None:
            raise Exception("LCRL expects MDP as an input")
        self.MDP = MDP
        if LDBA is None:
            raise Exception("LCRL expects LDBA as an input")
        self.LDBA = LDBA
        self.epsilon_transitions_exists = True if LDBA.epsilon_transitions.__len__() > 0 else False
        self.gamma = discount_factor
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.path_length = []
        self.Q = {}
        self.Q_initial_value = 0
        self.early_interruption = 0
        # ##### testing area ##### #
        self.test = False
        # ######################## #

    def train_ql(
            self, number_of_episodes,
            iteration_threshold,
            Q_initial_value=0
    ):
        self.MDP.reset()
        self.LDBA.reset()
        self.Q_initial_value = Q_initial_value

        if self.LDBA.accepting_sets is None:
            raise Exception('LDBA object is not defined properly. Please specify the "accepting_set". ')

        # product MDP: synchronise the MDP state with the automaton state
        current_state = self.MDP.current_state + [self.LDBA.automaton_state]
        product_MDP_action_space = self.MDP.action_space
        epsilon_transition_taken = False

        # check for epsilon-transitions at the current automaton state
        if self.epsilon_transitions_exists:
            product_MDP_action_space = self.action_space_augmentation()

        # initialise Q-value outside the main loop
        self.Q[str(current_state)] = {}
        for action_index in range(len(product_MDP_action_space)):
            self.Q[str(current_state)][product_MDP_action_space[action_index]] = Q_initial_value

        # main loop
        try:
            episode = 0
            self.path_length = [float("inf")]
            while episode < number_of_episodes:
                episode += 1
                self.MDP.reset()
                self.LDBA.reset()
                current_state = self.MDP.current_state + [self.LDBA.automaton_state]

                # check for epsilon-transitions at the current automaton state
                if self.epsilon_transitions_exists:
                    product_MDP_action_space = self.action_space_augmentation()

                Q_at_initial_state = []
                for action_index in range(len(product_MDP_action_space)):
                    Q_at_initial_state.append(self.Q[str(current_state)][product_MDP_action_space[action_index]])
                print('episode:' + str(episode) +
                      ', value function at s_0=' + str(max(Q_at_initial_state)) +
                      ', trace length=' + str(self.path_length[len(self.path_length) - 1]))
                iteration = 0
                path = current_state

                # each episode loop
                while iteration < iteration_threshold and \
                        self.LDBA.automaton_state != -1 and \
                        self.LDBA.accepting_frontier_set:
                    iteration += 1

                    # find the action with max Q at the current state
                    Qs = []
                    for action_index in range(len(product_MDP_action_space)):
                        Qs.append(self.Q[str(current_state)][product_MDP_action_space[action_index]])
                    maxQ_action_index = random.choice(np.where(Qs == np.max(Qs))[0])
                    maxQ_action = product_MDP_action_space[maxQ_action_index]

                    # check if an epsilon-transition is taken
                    if self.epsilon_transitions_exists and \
                            maxQ_action_index > len(self.MDP.action_space) - 1:
                        epsilon_transition_taken = True

                    # product MDP modification (for more details refer to https://bit.ly/LCRL_CDC_2019)
                    if epsilon_transition_taken:
                        next_MDP_state = self.MDP.current_state
                        next_automaton_state = self.LDBA.step(maxQ_action)
                    else:
                        # epsilon-greedy policy
                        if random.random() < self.epsilon:
                            next_MDP_state = self.MDP.step(random.choice(self.MDP.action_space))
                        else:
                            next_MDP_state = self.MDP.step(maxQ_action)
                        next_automaton_state = self.LDBA.step(self.MDP.state_label(next_MDP_state))

                    # product MDP: synchronise the automaton with MDP
                    next_state = next_MDP_state + [next_automaton_state]
                    if self.test:
                        print(str(maxQ_action) + ' | ' + str(next_state) + ' | ' + self.MDP.state_label(next_MDP_state))

                    # check for epsilon-transitions at the next automaton state
                    if self.epsilon_transitions_exists:
                        product_MDP_action_space = self.action_space_augmentation()

                    # Q values of the next state
                    Qs_prime = []
                    if str(next_state) not in self.Q.keys():
                        self.Q[str(next_state)] = {}
                        for action_index in range(len(product_MDP_action_space)):
                            self.Q[str(next_state)][product_MDP_action_space[action_index]] = Q_initial_value
                            Qs_prime.append(Q_initial_value)
                    else:
                        for action_index in range(len(product_MDP_action_space)):
                            Qs_prime.append(self.Q[str(next_state)][product_MDP_action_space[action_index]])

                    # update the accepting frontier set
                    if not epsilon_transition_taken:
                        reward_flag = self.LDBA.accepting_frontier_function(next_automaton_state)
                    else:
                        reward_flag = 0
                        epsilon_transition_taken = False

                    if reward_flag > 0:
                        state_dep_gamma = self.gamma
                    else:
                        state_dep_gamma = 1

                    # Q update
                    self.Q[str(current_state)][maxQ_action] = \
                        (1 - self.alpha) * self.Q[str(current_state)][maxQ_action] + \
                        self.alpha * (self.reward(reward_flag) + state_dep_gamma * np.max(Qs_prime))

                    # update the state
                    current_state = next_state
                    path.append(current_state)

                # check if at the end of episode the accepting frontier is full or not
                # for more details refer to https://bit.ly/LCRL_CDC_2019
                if self.LDBA.accepting_frontier_set:
                    self.path_length.append(float("inf"))
                else:
                    self.path_length.append(len(path))

        except KeyboardInterrupt:
            print('\nTraining exited early.')
            try:
                is_save = input(
                    'Would you like to save the training data? '
                    'If so, type in "y", otherwise, interrupt with CTRL+C. ')
            except KeyboardInterrupt:
                print('\nExiting...')

            if is_save == 'y' or is_save == 'Y':
                print('Saving...')
                self.early_interruption = 1

    def reward(self, reward_flag):
        if reward_flag > 0:
            return 1
        else:
            return 0

    def action_space_augmentation(self):
        if self.LDBA.automaton_state in self.LDBA.epsilon_transitions.keys():
            product_MDP_action_space = self.MDP.action_space + \
                                       self.LDBA.epsilon_transitions[self.LDBA.automaton_state]
        else:
            product_MDP_action_space = self.MDP.action_space
        return product_MDP_action_space
