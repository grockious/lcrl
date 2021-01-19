from lcrl.train import train
from lcrl.automata.goal1_then_goal2 import goal1_then_goal2
from lcrl.environments.gridworld_1 import gridworld_1

if __name__ == "__main__":
    MDP = gridworld_1
    LDBA = goal1_then_goal2
    train(MDP, LDBA, algorithm='ql', episode_num=2000, iteration_num_max=4000)

