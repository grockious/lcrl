from lcrl.automata.ldba import LDBA

# an example automaton for "goal1 then goal2 while avoiding unsafe" or "F (goal1 & XF (goal2)) & G !unsafe"
# automaton image is available in "./assets" or "https://i.imgur.com/R7Zg5mw.png"
# only the automaton "step" function and the "accepting_sets" attribute need to be specified.
# "accepting_sets" is a list of lists for Generalised Büchi Accepting condition (https://bit.ly/ldba_paper)
goal1_then_goal2 = LDBA(accepting_sets=[[2]])


# "step" function for the automaton transitions (input: label, output: automaton_state, un-accepting sink state is "-1")
def step(self, label):
    # state 0
    if self.automaton_state == 0:
        if 'goal1' in label and 'unsafe' not in label:
            self.automaton_state = 1
        elif 'unsafe' in label:
            self.automaton_state = -1  # un-accepting sink state
        else:
            self.automaton_state = 0
    # state 1
    elif self.automaton_state == 1:
        if 'goal2' in label and 'unsafe' not in label:
            self.automaton_state = 2
        elif 'unsafe' in label:
            self.automaton_state = -1  # un-accepting sink state
        else:
            self.automaton_state = 1
    # state 2
    elif self.automaton_state == 2:
        if 'unsafe' in label:
            self.automaton_state = -1  # un-accepting sink state
        else:
            self.automaton_state = 2
    # state -1
    elif self.automaton_state == -1:
        self.automaton_state = -1  # un-accepting sink state
    # step function returns the new automaton state
    return self.automaton_state


# now override the step function
LDBA.step = step.__get__(goal1_then_goal2, LDBA)

# finally, does the LDBA contains an epsilon transition? if so then
# for each state with outgoing epsilon-transition define a different epsilon
# example: <LDBA_object_name>.epsilon_transitions = {0: ['epsilon_0'], 4: ['epsilon_1']}
# "0" and "4" are automaton_states
# (for more details refer to https://bit.ly/ldba_paper)
