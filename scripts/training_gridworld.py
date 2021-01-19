# import train module
from lcrl.train import train
# either create an automata object or import built-in ones
from lcrl.automata.goal1_then_goal2 import goal1_then_goal2
# either create an environment object or import built-in ones
from lcrl.environments.gridworld_1 import gridworld_1

if __name__ == "__main__":
    MDP = gridworld_1
    LDBA = goal1_then_goal2
    # train module has the following inputs (
    #         MDP,
    #         LDBA,
    #         algorithm,
    #         episode_num,
    #         iteration_num_max,
    #         save_dir,
    #         discount_factor,
    #         learning_rate,
    #         epsilon,
    #         test,
    #         average_window,
    # )
    train(MDP, LDBA, algorithm='ql', episode_num=2500, iteration_num_max=4000)

