import numpy as np
import random


# TODO: Define an abstract class for the MDP and LDBA
class LCRL:
    """
    LCRL

    ...

    Attributes
    ----------
    MDP : an object from ./environments
        MDP object has to have the following properties
        (1) countably finite state and action spaces
        (2) a "step(action)" function describing the dynamics
        (3) a "state_label(state)" function that maps states to labels
        (4) a "reset()" function that resets the state to an initial state
        (5) current state of the MDP is "current_state"
        (6) action space is "action_space" and all actions are enabled in each state
    LDBA : an object from ./automata
        an automaton
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
            epsilon=0.15,
                ):
        if MDP is None:
            raise Exception("LCRL expects MDP as an input")
        self.MDP = MDP
        if LDBA is None:
            raise Exception("LCRL expects LDBA as an input")
        self.LDBA = LDBA
        self.epsilon_transitions_exists = 'epsilon' in self.LDBA.assignment.keys()
        self.gamma = discount_factor
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.path_length = []
        self.Q = {}
        self.Q_initial_value = 0
        # ##### testing area ##### #
        self.test = False

    def train_ql(
            self, number_of_episodes,
            iteration_threshold,
            Q_initial_value=0
                ):
        self.MDP.reset()
        self.LDBA.reset()
        self.Q_initial_value = Q_initial_value

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
                  ', value function at s_0='+str(max(Q_at_initial_state)) +
                  ', trace length='+str(self.path_length[len(self.path_length)-1]))
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
                        maxQ_action_index > len(self.MDP.action_space)-1:
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
                    print(str(maxQ_action)+' | '+str(next_state)+' | '+str(next_automaton_state))

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

                # Q update
                self.Q[str(current_state)][maxQ_action] = \
                    (1 - self.alpha) * self.Q[str(current_state)][maxQ_action] + \
                    self.alpha * (self.reward(reward_flag) + self.gamma * np.max(Qs_prime))

                # update the state
                current_state = next_state
                path.append(current_state)

            # check if at the end of episode the accepting frontier is full or not
            # for more details refer to https://bit.ly/LCRL_CDC_2019
            if self.LDBA.accepting_frontier_set:
                self.path_length.append(float("inf"))
            else:
                self.path_length.append(len(path))

    def reward(self, reward_flag):
        if reward_flag > 0:
            return 10
        elif reward_flag < 0:
            return -1
        else:
            return 0

    def action_space_augmentation(self):
        if self.LDBA.automaton_state in self.LDBA.assignment['epsilon'].keys():
            product_MDP_action_space = self.MDP.action_space + \
                self.LDBA.assignment['epsilon'][self.LDBA.automaton_state]
        else:
            product_MDP_action_space = self.MDP.action_space
        return product_MDP_action_space
