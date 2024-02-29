# import train module
from src.lcrl.train import train
# either create an automata object or import built-in ones
from src.lcrl.automata.mars_rover_1_3 import mars_rover_1_3
# either create an environment object or import built-in ones
from src.lcrl.environments.mars_rover_3_4 import mars_rover

if __name__ == "__main__":
    MDP = mars_rover
    LDBA = mars_rover_1_3
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
    task = train(MDP, LDBA, algorithm='ddpg',
                 episode_num=1000,
                 iteration_num_max=18000,
                 discount_factor=0.99,
                 learning_rate=0.05,
                 ddpg_replay_buffer_size=100000,
                 )
