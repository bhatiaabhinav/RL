import RL


class DQNContext(RL.Context):
    '''Most hyperparams are according to Rainbow DQN paper'''
    double_dqn = True
    dueling_dqn = True
    minibatch_size = 32
    experience_buffer_length = int(1e6)
    atari_framestack_k = 4
    atari_clip_rewards = True
    atari_episode_life = True
    target_network_update_every = 8000  # 8K Steps i.e 32K Frames
    target_network_update_tau = 1
    gamma = 0.99
    nsteps = 3
    atari_frameskip_k = 4
    train_every = 4
    gradient_steps = 1
    learning_rate = 6.25e-5
    adam_epsilon = 1e-8
    epsilon = 1
    final_epsilon = 0.01
    exploit_epsilon = 0.001
    epsilon_anneal_over = 62500  # 62.5K Steps or 250K Frames
    minimum_experience = 50000  # or 200K Frames. If using prioritized replay, use 20K Steps. i.e. 80K Frames
    atari_noop_max = 30
    convs = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    hidden_layers = [512]
    num_steps_to_run = int(5e7)  # 50M Steps. i.e. 200M Frames
    num_episodes_to_run = int(5e7)
    num_envs_to_make = 1
    exploit_every = 8  # episodes. They exploit every 250K Steps. That must be about 10 episodes.
