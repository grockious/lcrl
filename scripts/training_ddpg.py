# import train module
from src.train import train
# either create an automata object or import built-in ones
from src.automata.mars_rover_1_and_3 import mars_rover_1_and_3
# either create an environment object or import built-in ones
from src.environments.mars_rover_continuous_action import MarsRover

if __name__ == "__main__":
    MDP = MarsRover()
    LDBA = mars_rover_1_and_3
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
                 episode_num=80000,
                 iteration_num_max=50000,
                 discount_factor=0.9,
                 learning_rate=0.001,
                 ddpg_replay_buffer_size=50000,
                 )
