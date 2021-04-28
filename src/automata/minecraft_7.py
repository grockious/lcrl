from src.automata.ldba import LDBA

# an example automaton for "wood then iron then work_bench then gold" or
# "F (wood & XF (iron & XF (work_bench & XF gold)))"
# only the automaton "step" function and the "accepting_sets" attribute need to be specified
# "accepting_sets" for Generalised Büchi Accepting (more details here https://bit.ly/ldba_paper)
minecraft_7 = LDBA(accepting_sets=[[4]])


# "step" function for the automaton transitions (input: label, output: automaton_state, un-accepting sink state is "-1")
def step(self, label):
    # state 0
    if self.automaton_state == 0:
        if 'wood' in label:
            self.automaton_state = 1
        else:
            self.automaton_state = 0
    # state 1
    elif self.automaton_state == 1:
        if 'iron' in label:
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
        if 'gold' in label:
            self.automaton_state = 4
        else:
            self.automaton_state = 3
    # state 4
    elif self.automaton_state == 4:
        self.automaton_state = 4
    # step function returns the new automaton state
    return self.automaton_state


# now override the step function
LDBA.step = step.__get__(minecraft_7, LDBA)

# finally, does the LDBA contains an epsilon transition? if so then
# for each state with outgoing epsilon-transition define a different epsilon
# example: <LDBA_object>.epsilon_transitions = {0: ['epsilon_0'], 4: ['epsilon_1']}
# "0" and "4" are automaton_states
# (for more details refer to https://bit.ly/ldba_paper)
