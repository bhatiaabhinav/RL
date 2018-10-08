"""dqn.py: Implementation of Deepmind's DQN algorithm using Tensorflow and OpenAI gym.

original paper: https://www.nature.com/articles/nature14236

Usage:
    python3 dqn.py [env_id]
    #classic_control to know about more available environments.
    env_id: optional. default='CartPole-v0'. refer https://gym.openai.com/envs/

python dependencies:
    gym[classic_control], tensorflow, numpy

What has not been implemented yet:
    * Use of convolutional neural networks to learn atari environments from pixels.
    * Code for saving and loading the network.
"""

__author__ = "Abhinav Bhatia"
__email__ = "bhatiaabhinav93@gmail.com"
__license__ = "gpl"
__version__ = "0.9.0"

import os
import random
import sys
from collections import deque
from datetime import timedelta

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from RL.common import logger


class Experience:
    def __init__(self, state, action, reward, done, info, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.info = info
        self.next_state = next_state

    def __sizeof__(self):
        return sys.getsizeof(self.state) + sys.getsizeof(self.action) + \
            sys.getsizeof(self.reward) + sys.getsizeof(self.done) + \
            sys.getsizeof(self.info) + sys.getsizeof(self.next_state)


class ExperienceBuffer:
    '''A circular buffer to hold experiences'''

    def __init__(self, length=1e6, size_in_bytes=None):
        self.buffer = []  # type: List[Experience]
        self.buffer_length = length
        self.count = 0
        self.size_in_bytes = size_in_bytes
        self.next_index = 0

    def __len__(self):
        return self.count

    def add(self, exp: Experience):
        if self.count == 0:
            if self.size_in_bytes is not None:
                self.buffer_length = self.size_in_bytes / sys.getsizeof(exp)
            self.buffer_length = int(self.buffer_length)
            print('Initializing experience buffer of length {0}'.format(
                self.buffer_length))
            self.buffer = [None] * self.buffer_length
        self.buffer[self.next_index] = exp
        self.next_index = (self.next_index + 1) % self.buffer_length
        self.count = min(self.count + 1, self.buffer_length)

    def random_experiences(self, count):
        indices = np.random.randint(0, self.count, size=count)
        for i in indices:
            yield self.buffer[i]

    def random_states(self, count):
        experiences = list(self.random_experiences(count))
        return [e.state for e in experiences]


class LinearFrameStackWrapper(gym.Wrapper):
    def __init__(self, env, k=3):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        space = env.observation_space  # type: gym.spaces.Box
        assert len(space.shape) == 1  # can only stack 1-D frames
        self.observation_space = gym.spaces.Box(
            low=np.array(list(space.low) * k), high=np.array(list(space.high) * k))

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._observation()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        obs = np.concatenate(self.frames, axis=0)
        assert list(np.shape(obs)) == list(self.observation_space.shape)
        return obs


class MaxEpisodeStepsWrapper(gym.Wrapper):
    def __init__(self, env, max_steps=1000):
        super().__init__(env)
        self.max_steps = max_steps
        self._current_ep_steps = 0

    def reset(self):
        self._current_ep_steps = 0
        return self.env.reset()

    def step(self, action):
        obs, r, d, info = self.env.step(action)
        self._current_ep_steps += 1
        if self._current_ep_steps >= self.max_steps:
            d = True
        return obs, r, d, info


class Context:
    seed = 0
    gamma = 0.99
    default_env_id = 'CartPole-v0'
    layers = [256, 128]  # hidden layers
    n_episodes = 10000
    max_steps_per_episode = 600
    experience_buffer_length = int(1e6)
    minimum_experience = 1000  # in frames, before learning can begin
    final_epsilon = 0.1  # exploration rate in epislon-greedy action selection
    epsilon_anneal_over = 20000
    train_every = 4  # one sgd step every this many frames
    # target network updated every this many frames
    target_network_update_interval = 1000
    minibatch_size = 128
    learning_rate = 1e-4
    tau = 0.001  # ddpg style target network (soft) update step size
    # whether to use DDPG style target network update, or use original DQN style
    ddpg_target_network_update_mode = True
    experiment_name = 'dqn'
    double_dqn = False
    dueling_dqn = True
    l2_reg = 0
    clip_gradients = True
    video_interval = 50  # every these many episodes, video recording will be done
    exploit_every = 4  # every these many episodes, agent will play without exploration

    def __init__(self, env_id):
        self.end_id = env_id
        self.env = None  # type: gym.wrappers.Monitor
        self.session = None  # type: tf.Session
        logger.set_logdir(os.getenv('OPENAI_LOGDIR'), env_id, self.experiment_name)
        self.logdir = logger.get_dir()

    def make_env(self, env_id):
        self.env_id = env_id
        self.env = gym.make(self.env_id)
        self.env = self._wrappers(self.env)
        return self.env

    def _wrappers(self, env):
        # env = LinearFrameStackWrapper(env, 3)
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

    @property
    def epsilon(self):
        return self.final_epsilon + (1 - self.final_epsilon) * (1 - min(self.total_steps / self.epsilon_anneal_over, 1))

    @property
    def total_episodes(self):
        return len(self.env.stats_recorder.episode_rewards)

    @property
    def total_steps(self):
        return self.env.stats_recorder.total_steps

    @property
    def episode_steps(self):
        return self.env.stats_recorder.steps

    def print_stats(self, average_over=100, end='\n'):
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
        print('{0} : Ep_id: {1:04d}\tR: {2:05.1f} {3}\tAv_R: {4:06.2f}\tAv_Eval_R: {5:06.2f}\tL: {6:03d}\r'.format(
            str(t), ep_id, reward, ep_type, av_reward, eval_av_reward, length
        ), end=end)


class Brain:
    def __init__(self, context: Context, name):
        self.name = name
        self.context = context
        with tf.variable_scope(name):
            self._create_network(context)

    def _dense_net(self, inputs, hidden_layers, hidden_activation, output_size, output_activation, name):
        with tf.variable_scope(name):
            prev_layer = inputs
            for layer_id, layer in enumerate(hidden_layers):
                h = tf.layers.dense(prev_layer, layer,
                                    activation=None, name='h{0}'.format(layer_id))
                h = tc.layers.layer_norm(h, scale=True, center=True)
                h = hidden_activation(h)
                prev_layer = h
            output_layer = tf.layers.dense(
                prev_layer, output_size, name='output')
            output_layer = output_activation(output_layer)
        return output_layer

    def _create_network(self, context: Context, name='Q_network'):
        with tf.variable_scope(name):
            self._states_placeholder = tf.placeholder(dtype=context.env.observation_space.dtype, shape=[
                                                      None] + list(context.env.observation_space.shape), name='states')
            if not context.dueling_dqn:
                self._Q = self._dense_net(self._states_placeholder, context.layers,
                                          tf.nn.relu, context.env.action_space.n, lambda x: x, 'Q')
            else:
                self._A_dueling = self._dense_net(self._states_placeholder, context.layers,
                                                  tf.nn.relu, context.env.action_space.n, lambda x: x, 'A_dueling')
                self._V_dueling = self._dense_net(self._states_placeholder, context.layers,
                                                  tf.nn.relu, context.env.action_space.n, lambda x: x, 'V_dueling')
                self._Q = self._V_dueling + self._A_dueling - \
                    tf.reduce_mean(self._A_dueling, axis=1, keepdims=True)
            self._best_action_id = tf.argmax(self._Q, axis=1, name='action')
            self._V = tf.reduce_max(self._Q, axis=1, name='V')

    def get_Q(self, states):
        return self.context.session.run(self._Q, feed_dict={
            self._states_placeholder: states
        })

    def get_V(self, states):
        return self.context.session.run(self._V, feed_dict={
            self._states_placeholder: states
        })

    def get_action(self, states):
        return self.context.session.run(self._best_action_id, feed_dict={
            self._states_placeholder: states
        })

    def get_vars(self):
        my_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='{0}/Q_network'.format(self.name))
        return my_vars

    def get_trainable_vars(self):
        my_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/Q_network'.format(self.name))
        return my_vars

    def get_perturbable_vars(self):
        return [v for v in self.get_vars() if not('LayerNorm' in v.name or 'batch_norm' in v.name)]

    def setup_copy_to(self, other_brain, soft_copy=False, name='copy_brain_ops'):
        with tf.variable_scope(name):
            other_brain = other_brain  # type: Brain
            my_vars = self.get_vars()
            other_brain_vars = other_brain.get_vars()
            assert len(my_vars) == len(
                other_brain_vars), "Something is wrong! Both brains should have same number of vars"
            copy_ops = []
            for my_var, other_var in zip(my_vars, other_brain_vars):
                copy_op = tf.assign(other_var, my_var * self.context.tau + (
                    1 - self.context.tau) * other_var) if soft_copy else tf.assign(other_var, my_var)
                copy_ops.append(copy_op)
            return copy_ops

    def _get_l2_norm(self, _vars, exclusions=['output', 'bias'], name='l2_losses'):
        with tf.variable_scope(name):
            loss = 0
            for v in _vars:
                if not any(excl in v.name for excl in exclusions):
                    loss += tf.nn.l2_loss(v)
            return loss

    def setup_training(self, name='train_Q_network'):
        with tf.variable_scope(name):
            context = self.context
            self._desired_Q_placeholder = tf.placeholder(
                dtype='float32', shape=[None, context.env.action_space.n])
            error = tf.square(self._Q - self._desired_Q_placeholder)
            self._mse = tf.reduce_mean(error)
            trainable_vars = self.get_trainable_vars()
            loss = self._mse + context.l2_reg * \
                self._get_l2_norm(trainable_vars)
            optimizer = tf.train.AdamOptimizer(context.learning_rate)
            grads = tf.gradients(loss, trainable_vars)
            if context.clip_gradients:
                grads = [tf.clip_by_value(
                    grad, -1, 1) if 'output' in var.name else grad for grad, var in zip(grads, trainable_vars)]
            self._sgd_step = optimizer.apply_gradients(
                zip(grads, trainable_vars))

    def train(self, mb_states, mb_desiredQ):
        return self.context.session.run([self._sgd_step, self._mse], feed_dict={
            self._states_placeholder: mb_states,
            self._desired_Q_placeholder: mb_desiredQ
        })


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def dqn(env_id):
    context = Context(env_id)
    set_global_seeds(context.seed)
    env = context.make_env(env_id)  # type: gym.wrappers.Monitor
    env.seed(context.seed)
    main_brain = Brain(context, 'main_brain')
    main_brain.setup_training('main_brain/train_Q_network')
    target_brain = Brain(context, 'target_brain')
    target_brain_update_op = main_brain.setup_copy_to(
        target_brain, soft_copy=False, name='target_brain_update_op')
    target_brain_soft_update_op = main_brain.setup_copy_to(
        target_brain, soft_copy=True, name='target_brain_soft_update_op')

    with tf.Session() as session:
        context.session = session
        tf.summary.FileWriter(context.logdir, session.graph)
        session.run(tf.global_variables_initializer())
        session.run(target_brain_update_op)
        experience_buffer = ExperienceBuffer(
            length=context.experience_buffer_length)
        for episode_id in range(context.n_episodes):
            done = False
            eval_mode = context.should_eval_episode()
            env.stats_recorder.type = 'e' if eval_mode else 't'
            obs = env.reset()
            while not done:
                if np.random.random() > context.epsilon or eval_mode:
                    action = main_brain.get_action([obs])[0]
                else:
                    action = context.env.action_space.sample()
                obs1, r, done, info = env.step(action)
                experience = Experience(obs, action, r, done, info, obs1)
                experience_buffer.add(experience)
                # env.render()

                # let's train:
                frame_id = context.total_steps - 1
                if frame_id % context.train_every == 0 and frame_id > context.minimum_experience:
                    mb_exps = list(experience_buffer.random_experiences(
                        context.minibatch_size))
                    mb_states = [exp.state for exp in mb_exps]
                    mb_actions = [exp.action for exp in mb_exps]
                    mb_rewards = np.asarray([exp.reward for exp in mb_exps])
                    mb_dones = np.asarray([int(exp.done) for exp in mb_exps])
                    mb_states1 = [exp.next_state for exp in mb_exps]
                    if not context.double_dqn:
                        mb_states1_V = target_brain.get_V(mb_states1)
                    else:
                        mb_states1_a = main_brain.get_action(mb_states1)
                        mb_states1_Q = target_brain.get_Q(mb_states1)
                        mb_states1_V = np.asarray(
                            [mb_states1_Q[exp_id, mb_states1_a[exp_id]] for exp_id in range(context.minibatch_size)])
                    mb_gammas = (1 - mb_dones) * context.gamma
                    mb_desired_Q = main_brain.get_Q(mb_states)
                    for exp_id in range(context.minibatch_size):
                        mb_desired_Q[exp_id, mb_actions[exp_id]] = mb_rewards[exp_id] + \
                            mb_gammas[exp_id] * mb_states1_V[exp_id]
                    _, mse = main_brain.train(mb_states, mb_desired_Q)

                # update target network:
                if not context.ddpg_target_network_update_mode:
                    if frame_id % context.target_network_update_interval == 0:
                        session.run(target_brain_update_op)
                        print('Target network updated')
                else:
                    session.run(target_brain_soft_update_op)

                obs = obs1

            context.print_stats(
                average_over=100, end='\n' if episode_id % 50 == 0 else '\r')

        print('---------------Done--------------------')
        context.print_stats(average_over=context.n_episodes)


if __name__ == '__main__':
    env_id = sys.argv[1] if len(sys.argv) > 1 else Context.default_env_id
    dqn(env_id)
