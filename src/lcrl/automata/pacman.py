from lcrl.automata.ldba import LDBA

# an example automaton for "(food1 then food2) or (food2 then food1) while avoiding ghost" or
# "(F (food1 & F food2) || F (food2 & F food1)) & G !ghost"
# only the automaton "step" function and the "accepting_sets" attribute need to be specified.
# "accepting_sets" is a list of lists for Generalised Büchi Accepting condition (https://bit.ly/ldba_paper)
pacman = LDBA(accepting_sets=[[1], [2], [3]])


# "step" function for the automaton transitions (input: label, output: automaton_state, un-accepting sink state is "-1")
def step(self, label):
    # state 0
    if self.automaton_state == 0:
        if label is not None and 'food1' in label and 'ghost' not in label:
            self.automaton_state = 1
        elif label is not None and 'food2' in label and 'ghost' not in label:
            self.automaton_state = 2
        elif label is not None and 'ghost' in label:
            self.automaton_state = -1  # un-accepting sink state
        else:
            self.automaton_state = 0
    # state 1
    elif self.automaton_state == 1:
        if label is not None and 'food2' in label and 'ghost' not in label:
            self.automaton_state = 3
        elif label is not None and 'ghost' in label:
            self.automaton_state = -1  # un-accepting sink state
        else:
            self.automaton_state = 1
    # state 2
    elif self.automaton_state == 2:
        if label is not None and 'food1' in label and 'ghost' not in label:
            self.automaton_state = 3
        elif label is not None and 'ghost' in label:
            self.automaton_state = -1  # un-accepting sink state
        else:
            self.automaton_state = 2
    # state 3
    elif self.automaton_state == 3:
        if label is not None and 'ghost' in label:
            self.automaton_state = -1  # un-accepting sink state
        else:
            self.automaton_state = 3
    # state -1
    elif self.automaton_state == -1:
        self.automaton_state = -1  # un-accepting sink state
    # step function returns the new automaton state
    return self.automaton_state


# now override the step function
LDBA.step = step.__get__(pacman, LDBA)

# finally, does the LDBA contains an epsilon transition? if so then
# for each state with outgoing epsilon-transition define a different epsilon
# example: <LDBA_object_name>.epsilon_transitions = {0: ['epsilon_0'], 4: ['epsilon_1']}
# "0" and "4" are automaton_states
# (for more details refer to https://bit.ly/ldba_paper)
