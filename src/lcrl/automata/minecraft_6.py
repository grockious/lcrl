from lcrl.automata.ldba import LDBA

# an example automaton for "iron then wood then work_bench" or
# "F (iron & XF (wood & XF (work_bench)))"
# only the automaton "step" function and the "accepting_sets" attribute need to be specified.
# "accepting_sets" is a list of lists for Generalised Büchi Accepting condition (https://bit.ly/ldba_paper)
minecraft_6 = LDBA(accepting_sets=[[3]])


# "step" function for the automaton transitions (input: label, output: automaton_state, un-accepting sink state is "-1")
def step(self, label):
    # state 0
    if self.automaton_state == 0:
        if 'iron' in label:
            self.automaton_state = 1
        else:
            self.automaton_state = 0
    # state 1
    elif self.automaton_state == 1:
        if 'wood' in label:
            self.automaton_state = 2
        else:
            self.automaton_state = 1
    # state 2
    elif self.automaton_state == 2:
        if 'work_bench' in label:
            self.automaton_state = 3
        else:
            self.automaton_state = 2
    # state 3
    elif self.automaton_state == 3:
        self.automaton_state = 3
    # step function returns the new automaton state
    return self.automaton_state


# now override the step function
LDBA.step = step.__get__(minecraft_6, LDBA)

# finally, does the LDBA contains an epsilon transition? if so then
# for each state with outgoing epsilon-transition define a different epsilon
# example: <LDBA_object_name>.epsilon_transitions = {0: ['epsilon_0'], 4: ['epsilon_1']}
# "0" and "4" are automaton_states
# (for more details refer to https://bit.ly/ldba_paper)
