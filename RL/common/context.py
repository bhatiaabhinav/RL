import os
from datetime import timedelta

import gym
import numpy as np
import tensorflow as tf

from RL.common import logger
from RL.common.wrappers import MaxEpisodeStepsWrapper, LinearFrameStackWrapper  # noqa: F401
from RL.common.atari_wrappers import wrap_deepmind_with_framestack
from RL.common.summaries import Summaries  # noqa: F401


class Context:
    seed = 0
    gamma = 0.999
    # default_env_id = 'CartPole-v0'
    default_env_id = 'BreakoutNoFrameskip-v4'
    convs = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    states_embedding_hidden_layers = [400]
    Q_hidden_layers = [300]  # hidden layers
    n_episodes = 100000
    max_steps_per_episode = 600
    experience_buffer_length = int(1e6)
    minimum_experience = 1000  # in frames, before learning can begin
    final_epsilon = 0.1  # exploration rate in epislon-greedy action selection
    epsilon_anneal_over = 20000
    train_every = 4  # one sgd step every this many frames
    # target network updated every this many frames
    target_network_update_interval = 1000
    minibatch_size = 128  # 1/4th of this will be used for atari games
    nsteps = 8  # n-step TD learning
    learning_rate = 1e-4
    tau = 0.001  # ddpg style target network (soft) update step size
    # whether to use DDPG style target network update, or use original DQN style
    ddpg_target_network_update_mode = True
    experiment_name = 'dqn'
    double_dqn = False
    dueling_dqn = False
    l2_reg = 0
    clip_gradients = False
    clip_td_error = True
    video_interval = 50  # every these many episodes, video recording will be done
    exploit_every = 4  # every these many episodes, agent will play without exploration
    rnd_mode = False  # whether to use rnd intrinsic reward system
    rnd_num_features = 64  # predict these many random features of obs
    rnd_layers = [128]  # architecture of the rnd_predictor network
    rnd_learning_rate = 1e-4  # learning rate for the rnd_predictor network
    render = False

    def __init__(self, env_id):
        self.end_id = env_id
        self.env = None  # type: gym.wrappers.Monitor
        self.session = None  # type: tf.Session
        self.summaries = None  # type: Summaries
        self.intrinsic_returns = []
        logger.set_logdir(os.getenv('OPENAI_LOGDIR'),
                          env_id, self.experiment_name)
        self.logdir = logger.get_dir()

    def make_env(self, env_id):
        self.env_id = env_id
        self.env = gym.make(self.env_id)
        self.env = self._wrappers(self.env)
        assert self.minibatch_size % self.nsteps == 0, "mb size should be a multiple of nsteps"
        return self.env

    def _wrappers(self, env):
        # env = LinearFrameStackWrapper(env, 3)
        if self.need_conv_net:
            logger.log("Doing deepmind wrap")
            env = wrap_deepmind_with_framestack(env, episode_life=True, clip_rewards=False)
            logger.log("1/4th of the specified minibatch size will be used")
            self.minibatch_size = int(self.minibatch_size / 4)
        else:
            env = MaxEpisodeStepsWrapper(env, self.max_steps_per_episode)
        env = gym.wrappers.Monitor(env, os.path.join(
            self.logdir, 'monitor'), force=True, video_callable=self._video_schedule)
        return env

    def _video_schedule(self, episode_id):
        video_interval = self.video_interval
        from gym.wrappers.monitor import capped_cubic_video_schedule
        if video_interval is not None and video_interval <= 0:
            return False
        if episode_id == self.n_episodes - 1:
            return True
        if video_interval is None:
            return capped_cubic_video_schedule(episode_id)
        return episode_id % video_interval == 0

    def should_eval_episode(self, episode_id=None):
        if episode_id is None:
            episode_id = self.total_episodes
        return self.exploit_every is not None and episode_id % self.exploit_every == 0

    def activation_fn(self, x):
        return tf.nn.relu(x)

    @property
    def need_conv_net(self):
        return isinstance(self.env.observation_space, gym.spaces.Box) and len(self.env.observation_space.shape) >= 2

    @property
    def epsilon(self):
        return self.final_epsilon + (1 - self.final_epsilon) * (1 - min(self.total_steps / (self.epsilon_anneal_over + self.minimum_experience), 1))

    @property
    def total_episodes(self):
        return len(self.env.stats_recorder.episode_rewards)

    @property
    def total_steps(self):
        return self.env.stats_recorder.total_steps

    @property
    def episode_steps(self):
        return self.env.stats_recorder.steps

    def log_stats(self, average_over=100, end='\n'):
        np.set_printoptions(precision=2)
        ep_id = self.total_episodes - 1
        reward = self.env.stats_recorder.episode_rewards[-1]
        length = self.env.stats_recorder.episode_lengths[-1]
        ep_type = self.env.stats_recorder.episode_types[-1]
        av_reward = np.mean(
            self.env.stats_recorder.episode_rewards[-average_over:])
        eval_ep_rewards = [tup[1] for tup in filter(lambda tup: tup[0] == 'e', zip(
            self.env.stats_recorder.episode_types[-average_over:], self.env.stats_recorder.episode_rewards[-average_over:]))]
        t = self.env.stats_recorder.timestamps[-1] - \
            self.env.stats_recorder.initial_reset_timestamp
        t = timedelta(seconds=int(t))
        eval_av_reward = np.mean(eval_ep_rewards) if len(
            eval_ep_rewards) > 0 else 0
        intR = self.intrinsic_returns[-1]
        av_intR = np.mean(self.intrinsic_returns[-average_over:])
        logger.log('{0} : Ep_id: {1:04d}\tR: {2:05.1f} {3}\tAv_R: {4:06.2f}\tAv_Eval_R: {5:06.2f}\tL: {6:03d}\tep: {7:4.2f}\tintR: {8:05.1f}\tAv_intR: {9:06.2f}{10}'.format(
            str(t), ep_id, reward, ep_type, av_reward, eval_av_reward, length, self.epsilon, intR, av_intR, '\r' if end == '\r' else ''))
        if ep_id == 0:
            self.summaries.setup_scalar_summaries(["R", "R_exploit", "L", "intrinsic_R"])
        self.log_summary({"R": reward, "L": length, "intrinsic_R": intR}, ep_id)
        if ep_type == 'e':
            self.log_summary({"R_exploit": eval_ep_rewards[-1]}, ep_id)

    def log_summary(self, kvs, step):
        self.summaries.write_summaries(kvs, step)
