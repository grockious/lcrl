import numpy as np


# TODO: Define an abstract class for the LDBA
class LDBA:
    """
    Limit Deterministic Büchi Automaton (for more details refer to https://bit.ly/ldba_paper)

    ...

    Attributes
    ----------
    automaton_state : int
        initial automaton state (default 0)
    temporal_property : str
        the name of the linear temporal property (default 'g1-then-g2')

    Methods
    -------
    reset()
        resets the automaton state and its accepting frontier function
    step(label)
        changes the state of the automaton upon reading a label
    accepting_frontier_function(automaton_state)
        checks if the automaton state is in the accepting frontier set in order to update it
    """

    def __init__(self, automaton_state=0, temporal_property='g1-then-g2'):
        self.automaton_state = automaton_state
        self.temporal_property = temporal_property
        if temporal_property == 'g1-then-g2':
            self.assignment = {
                'safe': 0,
                'goal_1': 1,
                'goal_2': 2,
                'unsafe': -2,
                'obstacle': -1
            }
            self.accepting_frontier_set = [[1], [2]]  # the family set of accepting state sets (for more details refer to
            # https://bit.ly/LCRL_CDC_2019)
        elif temporal_property == 'g1-or-g2':
            self.assignment = {
                'epsilon': {0: ['epsilon_0', 'epsilon_1']},  # epsilon-transitions
                # for each state with outgoing epsilon-transition define a different epsilon
                # examples: 'epsilon': {automaton_state_1: ['epsilon_0'], automaton_state_4: ['epsilon_1']}
                # (for more details refer to https://bit.ly/ldba_paper)
                'safe': 0,
                'goal_1': 1,
                'goal_2': 2,
                'unsafe': -2,
                'obstacle': -1
            }
            self.accepting_frontier_set = [[1, 2]]  # the family set of accepting state sets (for more details refer to
            # https://bit.ly/LCRL_CDC_2019)
        else:
            raise NotImplementedError('Other temporal properties need to be defined')

    def reset(self):
        self.automaton_state = 0
        if self.temporal_property == 'g1-then-g2':
            self.accepting_frontier_set = [[1], [2]]  # the family set of accepting state sets (for more details refer to
        # https://bit.ly/LCRL_CDC_2019)
        elif self.temporal_property == 'g1-or-g2':
            self.accepting_frontier_set = [[1, 2]]  # the family set of accepting state sets (for more details refer to
        # https://bit.ly/LCRL_CDC_2019)

    def step(self, label):
        # LDBA 1
        if self.temporal_property == 'g1-then-g2':
            # state 0
            if self.automaton_state == 0:
                if label == self.assignment['goal_1']:
                    self.automaton_state = 1
                elif label == self.assignment['unsafe']:
                    self.automaton_state = -1  # un-accepting sink state
                else:
                    self.automaton_state = 0
            # state 1
            elif self.automaton_state == 1:
                if label == self.assignment['goal_2']:
                    self.automaton_state = 2  # accepting state
                elif label == self.assignment['unsafe']:
                    self.automaton_state = -1  # un-accepting sink state
                else:
                    self.automaton_state = 1
            # state 2
            elif self.automaton_state == 2:
                if label == self.assignment['unsafe']:
                    self.automaton_state = -1  # un-accepting sink state
                else:
                    self.automaton_state = 2  # accepting state

        # LDBA 2
        elif self.temporal_property == 'g1-or-g2':
            # state 0
            if self.automaton_state == 0:
                if label == self.assignment['epsilon'][self.automaton_state][0]:
                    self.automaton_state = 1
                elif label == self.assignment['epsilon'][self.automaton_state][1]:
                    self.automaton_state = 2
                elif label == self.assignment['unsafe']:
                    self.automaton_state = -1  # un-accepting sink state
                else:
                    self.automaton_state = 0
            # state 1
            elif self.automaton_state == 1:
                if label == self.assignment['goal_1']:
                    self.automaton_state = 1
                else:
                    self.automaton_state = -1  # un-accepting sink state
            # state 2
            elif self.automaton_state == 2:
                if label == self.assignment['goal_2']:
                    self.automaton_state = 2
                else:
                    self.automaton_state = -1  # un-accepting sink state

        # return automaton state
        return self.automaton_state

    def accepting_frontier_function(self, next_automaton_state):
        # for more details refer to https://bit.ly/LCRL_CDC_2019

        # remove the sets that have intersection with next_automaton_state
        indeces_to_remove = []
        for i in range(len(self.accepting_frontier_set)):
            if next_automaton_state in self.accepting_frontier_set[i]:
                indeces_to_remove.append(i)
        self.accepting_frontier_set = \
            np.delete(self.accepting_frontier_set, indeces_to_remove, axis=0).tolist()

        # return a positive flag if a set has been removed from the accepting frontier set
        if indeces_to_remove:
            return 1
        # return a negative flag if the non-accepting sink is visited
        elif next_automaton_state == -1:
            return -1
        # return zero otherwise
        else:
            return 0
