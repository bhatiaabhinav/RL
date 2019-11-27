import RL


class SACContext(RL.Context):
    num_critics = 2
    minibatch_size = 100
    experience_buffer_length = int(1e6)
    target_network_update_every = 1
    target_network_update_tau = 0.005
    gradient_steps = 1
    gamma = 0.99
    nsteps = 1
    train_every = 1
    learning_rate = 1e-3
    actor_learning_rate = 1e-4
    l2_reg = 0
    actor_l2_reg = 0
    adam_epsilon = 1e-8
    minimum_experience = 10000  # or 20K Frames. If using prioritized replay, use 2/5th number of steps.
    convs = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    hidden_layers = [256, 256]
    layer_norm = False
    normalize_observations = None
    normalize_actions = False
    init_scale = None
    num_steps_to_run = int(1e7)  # 10M Steps
    num_episodes_to_run = int(1e7)
    num_envs_to_make = 1
    exploit_every = 8  # episodes. They exploit every 250K Steps. That must be about 10 episodes.
    alpha = 0.2
    logstd_max = 2
    logstd_min = -20
