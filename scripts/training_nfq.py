# import train module
from lcrl.train import train
# either create an automata object or import built-in ones
from lcrl.automata.mars_rover_2_4 import mars_rover_2_4
# either create an environment object or import built-in ones
from lcrl.environments.mars_rover_1_2 import mars_rover

if __name__ == "__main__":
    # MDP = MarsRover()
    MDP = mars_rover
    LDBA = mars_rover_2_4
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
    task = train(MDP, LDBA, algorithm='nfq',
                 episode_num=50,
                 iteration_num_max=3000,
                 discount_factor=0.9,
                 learning_rate=0.01,
                 nfq_replay_buffer_size=3000,
                 )
