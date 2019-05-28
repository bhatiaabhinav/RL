import os
import sys
from typing import List  # noqa: F401

import numpy as np
import tensorflow as tf

import RL
from RL.common.summaries import Summaries
import gym
import roboschool  # noqa: F401


class Context:
    '''Object to hold hyperparameters and the envs and important shared variables. Functionality to parse command line args and read them into the hyperparameters'''
    default_env_id = 'CartPole-v0'
    default_experiment_name = 'untitled'
    default_logdir = os.getenv('OPENAI_LOGDIR')
    logger_level = 'DEBUG'
    seed = 0
    gamma = 0.99
    eval_mode = False
    convs = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    states_embedding_hidden_layers = []
    init_scale = None
    normalize_observations = True
    normalize_actions = False
    hidden_layers = [512]  # hidden layers
    layer_norm = False
    num_episodes_to_run = int(5e7)
    num_steps_to_run = int(5e7)
    num_envs_to_make = 1
    experience_buffer_length = int(1e6)
    experience_buffer_megabytes = None
    minimum_experience = 50000  # in frames, before learning can begin
    epsilon = 1  # for epislon-greedy action selection during exploration
    final_epsilon = 0.01  # anneal epsilon for epislon-greedy action selection during exploration to this value
    exploit_epsilon = 0.001  # for epislon-greedy action selection during exploitation
    epsilon_anneal_over = 62500
    param_noise_divergence = 0.2
    param_noise_adaptation_factor = 1.01
    train_every = 4  # gradient steps every this many steps
    gradient_steps = 1
    # target network updated every this many frames
    target_network_update_every = 4
    target_network_update_tau = 0.001  # ddpg style target network (soft) update step size
    minibatch_size = 32
    nsteps = 3  # n-step TD learning
    max_depth = 5  # max_depth for look ahead planning
    learning_rate = 1e-3
    adam_epsilon = 1e-8
    actor_learning_rate = 1e-4
    double_dqn = False
    dueling_dqn = False
    num_critics = 1
    l2_reg = 0
    actor_l2_reg = 0
    clip_gradients = False
    clip_td_error = False
    video_interval = 50  # every these many episodes, video recording will be done
    save_every = 100  # every these many episodes, save a model
    exploit_every = 8  # every these many episodes, agent will play without exploration
    rnd_mode = False  # whether to use rnd intrinsic reward system
    rnd_num_features = 64  # predict these many random features of obs
    rnd_layers = [128]  # architecture of the rnd_predictor network
    rnd_learning_rate = 1e-4  # learning rate for the rnd_predictor network
    np_print_precision = 3
    render = False
    render_mode = 'auto'
    auto_dispatch_on_render = True
    sensitivity_visualizer = False
    plot_Q = False
    render_interval = 1
    render_vsync = False
    pyglet_fps = -1
    safety_threshold = -0.1
    penalty_safe_dqn_multiplier = 10
    penalty_safe_dqn_mode = False
    safety_stream_names = ['Safety']
    load_model_dir = None
    load_every = None  # load the model every this many episodes
    atari_framestack_k = 4
    atari_frameskip_k = 4
    atari_noop_max = 30
    atari_clip_rewards = True
    atari_episode_life = True
    lunar_no_crash_penalty_main_stream = False
    alpha = 0.1
    beta = 0.1

    logstd_max = 2
    logstd_min = -20

    def activation_fn(self, x):
        return tf.nn.relu(x)

    @property
    def output_kernel_initializer(self):
        if self.init_scale is None:
            return None
        else:
            return tf.random_uniform_initializer(minval=-self.init_scale, maxval=self.init_scale)

    def __init__(self):
        self.env_id = self.default_env_id
        self.experiment_name = self.default_experiment_name
        self.logdir = self.default_logdir
        self.envs = []  # type: List[gym.Env]
        self._read_args()
        # numpy config
        np.set_printoptions(precision=self.np_print_precision)
        # tf session
        self.session = None  # type: tf.Session
        self.summaries = None  # type: Summaries
        # logdir config
        self.logdir = self._find_available_logdir(self.logdir, self.env_id, self.experiment_name)
        os.putenv('OPENAI_LOGDIR', self.logdir)
        RL.logger.reset()
        RL.logger.configure(self.logdir)
        RL.logger.set_level(getattr(RL.logger, self.logger_level))
        # save
        self._save()
        # misc
        RL.logger.info("env_id is ", self.env_id)
        if self.experiment_name == self.default_experiment_name:
            RL.logger.warn("No experiment_name argument was provided. It is highly recommened that you name your experiments")

    @property
    def env(self):
        return self.envs[0]

    def set_envs(self, envs: List[gym.Env]):
        self.envs.clear()
        self.envs.extend(envs)

    @property
    def num_envs(self):
        return len(self.envs)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def _read_args(self):
        for arg in sys.argv[1:]:
            words = arg.split('=')
            assert len(
                words) == 2, "Invalid argument {0}. The format is --arg_name=arg_value".format(arg)
            assert words[0].startswith(
                '--'), "Invalid argument {0}. The format is --arg_name=arg_value".format(arg)
            arg_name = words[0][2:]
            try:
                arg_value = eval(words[1])
            except Exception:
                arg_value = words[1]
            assert hasattr(self, arg_name), "Unrecognized argument {0}".format(arg_name)
            setattr(self, arg_name, arg_value)

    def _find_available_logdir(self, logdir_base, env, run_no_prefix):
        for run_no in range(int(1e6)):
            suffix = '_' + str(run_no).zfill(3) if run_no > 0 else ''
            logdir = os.path.join(logdir_base, env, run_no_prefix + suffix)
            if not os.path.isdir(logdir):
                return logdir
            else:
                run_no += 1

    def _save(self):
        with open(os.path.join(RL.logger.get_dir(), "context_class.json"), "w") as f:
            v = vars(Context)
            f.writelines(str(v))
        with open(os.path.join(RL.logger.get_dir(), "context.json"), "w") as f:
            v = vars(self)
            f.writelines(str(v))
