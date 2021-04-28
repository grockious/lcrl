import numpy as np


class LDBA:
    """
    Limit Deterministic Büchi Automaton (for more details refer to https://bit.ly/ldba_paper)

    ...

    Attributes
    ----------
    initial_automaton_state : int
        initial automaton state (default 0)
    accepting_sets : list
        the set of accepting sets for Generalised Büchi Accepting condition (more details here https://bit.ly/ldba_paper)

    Methods
    -------
    reset()
        resets the automaton state and its accepting frontier function
    step(label)
        changes and returns the state of the automaton (self.automaton_state) upon reading a label (un-accepting sink state is "-1")
    accepting_frontier_function(automaton_state)
        checks if the automaton state is in the accepting frontier set in order to update it
    """

    def __init__(self, initial_automaton_state=0, accepting_sets=None):
        self.initial_automaton_state = initial_automaton_state
        self.automaton_state = self.initial_automaton_state
        self.accepting_frontier_set = accepting_sets
        self.accepting_sets = accepting_sets
        self.epsilon_transitions = {}

    def reset(self):
        self.automaton_state = self.initial_automaton_state
        self.accepting_frontier_set = self.accepting_sets.copy()

    def step(self, label):
        pass

    def accepting_frontier_function(self, next_automaton_state):
        # for more details refer to the tool paper

        # remove the sets that have intersection with next_automaton_state
        indeces_to_remove = []
        for i in range(len(self.accepting_frontier_set)):
            if next_automaton_state in self.accepting_frontier_set[i]:
                indeces_to_remove.append(i)
        self.accepting_frontier_set = \
            np.delete(self.accepting_frontier_set, indeces_to_remove, axis=0).tolist()

        if indeces_to_remove and not self.accepting_frontier_set:
            self.accepting_frontier_set = self.accepting_sets.copy()
            self.accepting_frontier_set = \
                np.delete(self.accepting_frontier_set, indeces_to_remove, axis=0).tolist()

        # return a positive flag if a set has been removed from the accepting frontier set
        if indeces_to_remove:
            return 1
        # return zero otherwise
        else:
            return 0
