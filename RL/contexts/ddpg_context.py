import RL


class DDPGContext(RL.Context):
    num_critics = 1
    minibatch_size = 64
    experience_buffer_length = int(1e6)
    target_network_update_every = 1  # 8K Steps i.e 32K Frames
    target_network_update_tau = 0.001
    gamma = 0.99
    nsteps = 1
    train_every = 1
    learning_rate = 1e-3
    actor_learning_rate = 1e-4
    l2_reg = 1e-2
    actor_l2_reg = 0
    adam_epsilon = 1e-8
    minimum_experience = 10000  # or 20K Frames. If using prioritized replay, use 2/5th number of steps.
    convs = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    hidden_layers = [400, 300]
    layer_norm = False
    normalize_observations = True
    normalize_actions = False
    init_scale = 3e-3
    num_steps_to_run = int(1e7)  # 10M Steps
    num_episodes_to_run = int(1e7)
    num_envs_to_make = 1
    exploit_every = 8  # episodes. They exploit every 250K Steps. That must be about 10 episodes.
    param_noise_divergence = 0.2
    param_noise_adaptation_factor = 1.01
