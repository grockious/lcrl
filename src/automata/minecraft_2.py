from src.automata.ldba import LDBA

# an example automaton for "grass then tool_shed" or "F (wood & XF (tool_shed))"
# only the automaton "step" function and the "accepting_sets" attribute need to be specified
# "accepting_sets" for Generalised Büchi Accepting (more details here https://bit.ly/ldba_paper)
minecraft_2 = LDBA(accepting_sets=[[2]])


# "step" function for the automaton transitions (input: label, output: automaton_state, un-accepting sink state is "-1")
def step(self, label):
    # state 0
    if self.automaton_state == 0:
        if 'grass' in label:
            self.automaton_state = 1
        else:
            self.automaton_state = 0
    # state 1
    elif self.automaton_state == 1:
        if 'tool_shed' in label:
            self.automaton_state = 2
        else:
            self.automaton_state = 1
    # state 2
    elif self.automaton_state == 2:
        self.automaton_state = 2
    # step function returns the new automaton state
    return self.automaton_state


# now override the step function
LDBA.step = step.__get__(minecraft_2, LDBA)

# finally does the LDBA contains an epsilon transition? if so then
# for each state with outgoing epsilon-transition define a different epsilon
# example: <LDBA_object>.epsilon_transitions = {0: ['epsilon_0'], 4: ['epsilon_1']}
# "0" and "4" are automaton_states
# (for more details refer to https://bit.ly/ldba_paper)
