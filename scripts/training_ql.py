# import train module
from src.train import train
# either create an automata object or import built-in ones
from src.automata.minecraft_1 import minecraft_1
# either create an environment object or import built-in ones
from src.environments.minecraft import minecraft

if __name__ == "__main__":
    MDP = minecraft
    LDBA = minecraft_1
    # train module has the following inputs (
    #         MDP,
    #         LDBA,
    #         algorithm,
    #         episode_num,
    #         iteration_num_max,
    #         discount_factor,
    #         learning_rate,
    #         nfq_replay_buffer_size,
    #         decaying_learning_rate,
    #         epsilon,
    #         save_dir,
    #         test,
    #         average_window,
    # )
    task = train(MDP, LDBA,
                 algorithm='ql',
                 episode_num=500,
                 iteration_num_max=4000,
                 discount_factor=0.95,
                 learning_rate=0.9
                 )
