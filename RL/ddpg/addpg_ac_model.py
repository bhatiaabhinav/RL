import os

import gym
import joblib
import numpy as np
import tensorflow as tf
from gym.spaces import Box

from RL.common import logger
from RL.reco_rl.constraints import (count_nodes_in_constraints,
                                    depth_of_constraints, tf_infeasibility,
                                    tf_nested_constrained_softmax)
from RL.reco_rl.optnet import tf_optnet_layer

from RL.common.utils import (tf_deep_net,  # tf_safe_softmax_with_non_uniform_individual_constraints,; tf_softmax_with_min_max_constraints,
                             tf_log_transform_adaptive,
                             tf_normalize, tf_scale)


class RunningStats:

    def __init__(self, session: tf.Session, shape, epsilon=1e-2):
        self.n = 0
        self.old_m = np.zeros(shape=shape)
        self.new_m = np.zeros(shape=shape)
        self.old_s = np.zeros(shape=shape)
        self.new_s = np.zeros(shape=shape)
        self.session = session
        self.shape = shape
        self.epsilon = epsilon
        self.mean = tf.get_variable(dtype=tf.float32, shape=shape, initializer=tf.constant_initializer(
            0.0), name="running_mean", trainable=False)
        self.std = tf.get_variable(dtype=tf.float32, shape=shape, initializer=tf.constant_initializer(
            epsilon), name="running_std", trainable=False)

    def setup_update(self):
        self._mean_placeholder = tf.placeholder(
            dtype=tf.float32, shape=self.shape, name="running_mean_placeholder")
        self._std_placeholder = tf.placeholder(
            dtype=tf.float32, shape=self.shape, name="running_std_placeholer")
        self._set_mean = tf.assign(self.mean, self._mean_placeholder)
        self._set_std = tf.assign(self.std, self._std_placeholder)

    def update(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = np.zeros(shape=self.shape)
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

        mean = self.new_m if self.n else np.zeros(shape=self.shape)
        variance = self.new_s / \
            (self.n - 1) if self.n > 1 else np.zeros(shape=self.shape)
        variance = np.maximum(variance, self.epsilon)
        std = np.sqrt(variance)

        self.session.run([self._set_mean, self._set_std], feed_dict={
            self._mean_placeholder: mean,
            self._std_placeholder: std
        })


class DDPG_Model_Base:
    def __init__(self, session: tf.Session, name, env: gym.Env, ob_space: Box, ac_space: Box, constraints, softmax_actor,
                 soft_constraints, cp_optnet, nn_size, init_scale, advantage_learning, use_layer_norm, use_batch_norm,
                 use_norm_actor, log_norm_obs_alloc, log_norm_action, rms_norm_action):
        assert len(
            ac_space.shape) == 1, "Right now only flat action spaces are supported"
        assert (not softmax_actor) or (
            not soft_constraints), "Cannot use both soft and hard constraints"
        self.session = session
        self.name = name
        self.env = env
        self.ac_shape = ac_space.shape
        self.ac_low = ac_space.low
        self.ac_high = ac_space.high
        self.constraints = constraints
        self.ob_shape = ob_space.shape
        self.ob_dtype = ob_space.dtype
        self.ob_low = ob_space.low
        self.ob_high = ob_space.high
        self.nn_size = nn_size
        self.init_scale = init_scale
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_norm_actor = use_norm_actor
        self.log_norm_obs_alloc = log_norm_obs_alloc
        self.log_norm_action = log_norm_action
        self.rms_norm_action = rms_norm_action
        self.softmax_actor = softmax_actor
        self.soft_constraints = soft_constraints
        self.cp_optnet = cp_optnet
        self.advantage_learning = advantage_learning
        self.DUMMY_ACTION = [np.zeros(self.ac_shape)]

    def _setup_states_feed(self):
        self._states_feed = tf.placeholder(dtype=self.ob_dtype, shape=[
                                           None] + list(self.ob_shape), name="states_feed")

    def _setup_running_ob_stats(self):
        with tf.variable_scope('model/running_ob_stats'):
            self._ob_stats = RunningStats(self.session, self.ob_shape)

    def _setup_running_ac_stats(self):
        with tf.variable_scope('model/running_ac_stats'):
            self._ac_stats = RunningStats(self.session, self.ac_shape)

    def _tf_normalize_states(self, states, scope, is_training):
        with tf.variable_scope(scope):
            if self.log_norm_obs_alloc:
                logger.log(
                    'Using log normalization for allocation part of observation. rms_norm for rest of it.')
                states = tf_scale(states, self.ob_low,
                                  self.ob_high, 0, 1, 'scale_0_to_1_states')
                zones = self.ac_shape[0]
                states_feed_demand = states[:, :-zones - 1]
                states_feed_alloc = states[:, -zones - 1:-1]
                states_feed_time = states[:, -1:]
                states_feed_demand = tf_normalize(
                    states_feed_demand, self._ob_stats.mean[:-zones - 1], self._ob_stats.std[:-zones - 1], 'rms_norm_demand')
                states_feed_alloc = tf_log_transform_adaptive(
                    states_feed_alloc, 'log_norm_alloc', uniform_gamma=True)
                states_feed_alloc = tf_scale(
                    states_feed_alloc, 0, 1, -1, 1, 'scale_minus_1_to_1_alloc')
                states_feed_time = tf_normalize(
                    states_feed_time, self._ob_stats.mean[-1:], self._ob_stats.std[-1:], 'rms_norm_time')
                states = tf.concat(
                    [states_feed_demand, states_feed_alloc, states_feed_time], axis=-1, name='states_concat')
            else:
                logger.log('Using rms_norm for observation normalization')
                states = tf_normalize(
                    states, self._ob_stats.mean, self._ob_stats.std, 'rms_norm_states')
            return states

    def _tf_normalized_actions(self, actions, scope, is_training):
        with tf.variable_scope(scope):
            if self.log_norm_action:
                actions = tf_scale(actions, self.ac_low,
                                   self.ac_high, 0, 1, 'scale_0_to_1_actions')
                actions = tf_log_transform_adaptive(
                    actions, scope='log_norm_actions', uniform_gamma=True)
                actions = tf_scale(actions, 0, 1, -1, 1,
                                   'scale_minus_1_to_1_actions')
            elif self.rms_norm_action:
                actions = tf_normalize(
                    actions, self._ac_stats.mean, self._ac_stats.std, 'rms_norm')
            else:
                '''do not transform actions'''
            return actions

    def _setup_actor(self):
        with tf.variable_scope('model/actor'):
            self._is_training_a = tf.placeholder(
                dtype=tf.bool, name='is_training_a')
            states = self._tf_normalize_states(
                self._states_feed, 'normalized_states', self._is_training_a)
            output_shape = [count_nodes_in_constraints(
                self.constraints) - 1] if self.softmax_actor else self.ac_shape
            a = tf_deep_net(states, self.ob_shape, self.ob_dtype, 'a_network', self.nn_size, use_ln=self.use_layer_norm and self.use_norm_actor,
                            use_bn=self.use_batch_norm and self.use_norm_actor, training=self._is_training_a, output_shape=output_shape)
            self._penalizable_a = None
            if self.softmax_actor:
                logger.log(
                    'Actor output using nested contrained softmax layer')
                z_tree = {}
                a = tf_nested_constrained_softmax(
                    a, self.constraints, 'nested_constrained_softmax', z_tree=z_tree)
                with open(os.path.join(logger.get_dir(), 'z_tree.txt'), 'w') as file:
                    file.write(str(z_tree))
            elif self.soft_constraints and not self.cp_optnet:
                logger.log('Actor output in range 0 to 1 using scaled tanh')
                a = tf.minimum(tf.maximum(
                    (tf.nn.tanh(a, 'tanh') + 1) / 2, 0), 1)
                self._penalizable_a = a
            elif self.cp_optnet:
                assert depth_of_constraints(
                    self.constraints) == 1, "Right now only 1 level constraints are supported in optnet"
                c = self.constraints['equals']
                cmin = np.array([child['min']
                                 for child in self.constraints['children']])
                cmax = np.array([child['max']
                                 for child in self.constraints['children']])
                r = self.env.metadata.get('nresources', 1)
                logger.log(
                    "Diving the neural network output by {0} so that it need not be in small fractions".format(r))
                a = a / r
                if self.soft_constraints:
                    self._penalizable_a = a
                logger.log('Actor output using optnet')
                a = tf_optnet_layer(
                    a, c, cmin, cmax, len(cmin), scale_inputs=True)
            else:
                logger.log('Actor output in range -1 to 1 using tanh')
                a = tf.nn.tanh(a, 'tanh')
            self._use_actions_feed = tf.placeholder(
                dtype=tf.bool, name='use_actions_feed')
            self._actions_feed = tf.placeholder(
                dtype=tf.float32, shape=[None] + list(self.ac_shape), name='actions_feed')
            self._a = tf.case([
                (self._use_actions_feed, lambda: self._actions_feed)
            ], default=lambda: a)

    def _setup_critic(self):
        with tf.variable_scope('model/critic'):
            self._is_training_critic = tf.placeholder(
                dtype=tf.bool, name='is_training_critic')
            with tf.variable_scope('A'):
                states = self._tf_normalize_states(
                    self._states_feed, 'normalized_states', self._is_training_critic)
                s_after_one_hidden = tf_deep_net(states, self.ob_shape, self.ob_dtype, 'one_hidden',
                                                 self.nn_size[0:1], use_ln=self.use_layer_norm, use_bn=self.use_batch_norm, training=self._is_training_critic, output_shape=None)
                a = self._tf_normalized_actions(
                    self._a, 'normalized_actions', self._is_training_critic)
                s_a_concat = tf.concat(
                    [s_after_one_hidden, a], axis=-1, name="s_a_concat")
                self._A = tf_deep_net(s_a_concat, [self.nn_size[0] + self.ac_shape[0]], 'float32', 'A_network',
                                      self.nn_size[1:], use_ln=self.use_layer_norm, use_bn=self.use_batch_norm, training=self._is_training_critic, output_shape=[1])[:, 0]
            if self.advantage_learning:
                states = self._tf_normalize_states(
                    self._states_feed, 'normalized_states', self._is_training_critic)
                self._V = tf_deep_net(states, self.ob_shape, self.ob_dtype, 'V', self.nn_size, use_ln=self.use_layer_norm,
                                      use_bn=self.use_batch_norm, training=self._is_training_critic, output_shape=[1])[:, 0]
                self._Q = tf.add(self._V, self._A, name='Q')
            else:
                self._Q = self._A

    def _get_tf_variables(self, extra_scope=''):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, '{0}/model/{1}'.format(self.name, extra_scope))

    def _get_tf_trainable_variables(self, extra_scope=''):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{0}/model/{1}'.format(self.name, extra_scope))

    def _get_tf_perturbable_variables(self, extra_scope=''):
        return [var for var in self._get_tf_variables(extra_scope) if not('LayerNorm' in var.name or 'batch_norm' in var.name or 'log_norm' in var.name or 'running_ob_stats' in var.name or 'running_ac_stats' in var.name)]

    def _get_update_ops(self, extra_scope=''):
        return tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='{0}/model/{1}'.format(self.name, extra_scope))

    def get_a(self, states):
        return self.session.run(self._a, feed_dict={
            self._states_feed: states,
            self._is_training_a: False,
            self._use_actions_feed: False,
            self._actions_feed: self.DUMMY_ACTION,
        })

    def get_a_V_A_Q(self, states):
        return self.session.run([self._a, self._V, self._A, self._Q], feed_dict={
            self._states_feed: states,
            self._actions_feed: self.DUMMY_ACTION,
            self._is_training_a: False,
            self._use_actions_feed: False,
            self._is_training_critic: False
        })

    def get_a_Q(self, states):
        return self.session.run([self._a, self._Q], feed_dict={
            self._states_feed: states,
            self._actions_feed: self.DUMMY_ACTION,
            self._is_training_a: False,
            self._use_actions_feed: False,
            self._is_training_critic: False
        })

    def get_V_A_Q(self, states, actions):
        return self.session.run([self._V, self._A, self._Q], feed_dict={
            self._states_feed: states,
            self._use_actions_feed: True,
            self._actions_feed: actions,
            self._is_training_a: False,
            self._is_training_critic: False
        })

    def get_Q(self, states, actions):
        return self.session.run([self._Q], feed_dict={
            self._states_feed: states,
            self._use_actions_feed: True,
            self._actions_feed: actions,
            self._is_training_a: False,
            self._is_training_critic: False
        })

    def Q(self, s, a):
        return self.get_Q(s, a)

    def max_Q(self, s):
        return self.get_a_Q(s)[1]

    def V(self, s):
        return self.get_a_V_A_Q(s)[1]

    def A(self, s, a):
        return self.get_V_A_Q(s, a)[1]

    def max_A(self, s):
        return self.get_a_V_A_Q(s)[2]

    def argmax_Q(self, s):
        return self.get_a(s)


class DDPG_Model_Main(DDPG_Model_Base):
    def __init__(self, session: tf.Session, name, env: gym.Env, ob_space: Box, ac_space: Box, constraints, softmax_actor,
                 soft_constraints, soft_constraints_lambda, cp_optnet, nn_size, lr, a_lr, init_scale, advantage_learning,
                 use_layer_norm, use_batch_norm, use_norm_actor, l2_reg, a_l2_reg, clip_norm,
                 a_clip_norm, log_norm_obs_alloc, log_norm_action, rms_norm_action, **kwargs):
        super().__init__(session=session, name=name, env=env, ob_space=ob_space, ac_space=ac_space, constraints=constraints, softmax_actor=softmax_actor,
                         soft_constraints=soft_constraints, cp_optnet=cp_optnet, nn_size=nn_size, init_scale=init_scale,
                         advantage_learning=advantage_learning, use_layer_norm=use_layer_norm,
                         use_batch_norm=use_batch_norm, use_norm_actor=use_norm_actor,
                         log_norm_obs_alloc=log_norm_obs_alloc, log_norm_action=log_norm_action, rms_norm_action=rms_norm_action)
        logger.log("Setting up main network")
        with tf.variable_scope(name):
            self._setup_states_feed()
            self._setup_running_ob_stats()
            self._setup_running_ob_stats_update()
            self._setup_running_ac_stats()
            self._setup_running_ac_stats_update()
            self._setup_actor()
            self._setup_critic()
            self._setup_training(a_lr=a_lr, a_l2_reg=a_l2_reg, a_clip_norm=a_clip_norm,
                                 lr=lr, l2_reg=l2_reg, clip_norm=clip_norm, soft_constraints_lambda=soft_constraints_lambda)
            self._setup_saving_loading_ops()

    def _setup_running_ob_stats_update(self):
        with tf.variable_scope('update_running_ob_stats'):
            self._ob_stats.setup_update()

    def _setup_running_ac_stats_update(self):
        with tf.variable_scope('update_running_ac_stats'):
            self._ac_stats.setup_update()

    def _setup_actor_training(self, a_l2_reg, a_clip_norm, soft_constraints_lambda):
        with tf.variable_scope('optimize_actor'):
            # for training actions: maximize Advantage i.e. A
            self._a_vars = self._get_tf_trainable_variables('actor')
            self._av_A = tf.reduce_mean(self._A)
            self._infeasibility = tf.constant(0.0)
            if self.constraints is not None:
                with tf.name_scope('infeasibility'):
                    a_to_penalize = self._a if self._penalizable_a is None else self._penalizable_a
                    sum_violation, min_violation, max_violation = tf_infeasibility(
                        a_to_penalize, self.constraints, 'infeasibility_{0}'.format(self.constraints['name']))
                    self._infeasibility = tf.reduce_mean(
                        sum_violation + min_violation + max_violation)
            if self.soft_constraints:
                logger.log('Adding infeasibility penalty with lambda {0}'.format(
                    soft_constraints_lambda))
                loss = -self._av_A + soft_constraints_lambda * self._infeasibility
            else:
                loss = -self._av_A
            if a_l2_reg > 0:
                with tf.name_scope('L2_Losses'):
                    l2_loss = 0
                    for var in self._a_vars:
                        if 'bias' not in var.name and 'output' not in var.name:
                            l2_loss += a_l2_reg * tf.nn.l2_loss(var)
                loss = loss + l2_loss
            update_ops = self._get_update_ops('actor')
            with tf.control_dependencies(update_ops):
                a_grads = tf.gradients(loss, self._a_vars)
                if a_clip_norm is not None:
                    a_grads = [tf.clip_by_norm(
                        grad, clip_norm=a_clip_norm) for grad in a_grads]
                self._train_a_op = self._optimizer_a.apply_gradients(
                    list(zip(a_grads, self._a_vars)))
                # self.train_a_op = optimizer_a.minimize(-self.av_A, var_list=self.a_vars)

            # for supervised training of actions:
            self._a_desired = tf.placeholder(dtype=tf.float32, shape=[
                                             None] + list(self.ac_shape), name='desired_actions_feed')
            se = tf.square(self._a - self._a_desired)
            self._a_mse = tf.reduce_mean(se)
            loss = self._a_mse
            if a_l2_reg > 0:
                loss = loss + l2_loss
            with tf.control_dependencies(update_ops):
                a_grads = tf.gradients(loss, self._a_vars)
                if a_clip_norm is not None:
                    a_grads = [tf.clip_by_norm(
                        grad, clip_norm=a_clip_norm) for grad in a_grads]
                self._train_a_supervised_op = self._optimizer_a_supervised.apply_gradients(
                    list(zip(a_grads, self._a_vars)))

    def _setup_critic_training(self, l2_reg, clip_norm):
        if self.advantage_learning:
            with tf.variable_scope('optimize_V'):
                # for training V:
                self._V_vars = self._get_tf_trainable_variables('critic/V')
                self._V_target_feed = tf.placeholder(
                    dtype='float32', shape=[None])
                se = tf.square(self._V - self._V_target_feed)
                self._V_mse = tf.reduce_mean(se)
                loss = self._V_mse
                if l2_reg > 0:
                    with tf.variable_scope('L2_Losses'):
                        l2_loss = 0
                        for var in self._V_vars:
                            if 'bias' not in var.name and 'output' not in var.name:
                                l2_loss += l2_reg * tf.nn.l2_loss(var)
                        loss = loss + l2_loss
                update_ops = self._get_update_ops('critic/V')
                with tf.control_dependencies(update_ops):
                    V_grads = tf.gradients(loss, self._V_vars)
                    if clip_norm is not None:
                        V_grads = [tf.clip_by_norm(
                            grad, clip_norm=clip_norm) for grad in V_grads]
                    self._train_V_op = self._optimizer_V.apply_gradients(
                        list(zip(V_grads, self._V_vars)))
                    # self.train_V_op = optimizer_q.minimize(self.V_mse, var_list=self.V_vars)

        with tf.variable_scope('optimize_A'):
            # for training A:
            self._A_vars = self._get_tf_trainable_variables('critic/A')
            self._A_target_feed = tf.placeholder(dtype='float32', shape=[None])
            se = tf.square(self._A - self._A_target_feed)
            self._A_mse = tf.reduce_mean(se)
            loss = self._A_mse
            if l2_reg > 0:
                with tf.variable_scope('L2_Losses'):
                    l2_loss = 0
                    for var in self._A_vars:
                        if 'bias' not in var.name and 'output' not in var.name:
                            l2_loss += l2_reg * tf.nn.l2_loss(var)
                loss = loss + l2_loss
            update_ops = self._get_update_ops('critic/A')
            with tf.control_dependencies(update_ops):
                A_grads = tf.gradients(loss, self._A_vars)
                if clip_norm is not None:
                    A_grads = [tf.clip_by_norm(
                        grad, clip_norm=clip_norm) for grad in A_grads]
                self._train_A_op = self._optimizer_A.apply_gradients(
                    list(zip(A_grads, self._A_vars)))
                # self.train_A_op = optimizer_q.minimize(self.A_mse, var_list=self.A_vars)

    def _setup_training(self, a_lr, a_l2_reg, a_clip_norm, lr, l2_reg, clip_norm, soft_constraints_lambda):
        with tf.variable_scope('training'):
            with tf.variable_scope('optimizers'):
                self._optimizer_A = tf.train.AdamOptimizer(
                    learning_rate=lr, name='A_adam')
                self._optimizer_V = tf.train.AdamOptimizer(
                    learning_rate=lr, name='V_adam')
                self._optimizer_a = tf.train.AdamOptimizer(
                    learning_rate=a_lr, name='actor_adam')
                self._optimizer_a_supervised = tf.train.AdamOptimizer(
                    learning_rate=a_lr, name='actor_supervised_adam')
            self._setup_actor_training(
                a_l2_reg, a_clip_norm, soft_constraints_lambda)
            self._setup_critic_training(l2_reg, clip_norm)

    def _setup_saving_loading_ops(self):
        with tf.variable_scope('saving_loading_ops'):
            # for saving and loading
            params = self._get_tf_variables()
            self._load_placeholders = []
            self._load_ops = []
            for p in params:
                p_placeholder = tf.placeholder(
                    shape=p.shape.as_list(), dtype=tf.float32)
                self._load_placeholders.append(p_placeholder)
                self._load_ops.append(p.assign(p_placeholder))

    def update_running_ob_stats(self, obs):
        self._ob_stats.update(obs)

    def update_running_ac_stats(self, action):
        self._ac_stats.update(action)

    def train_V(self, states, target_V):
        return self.session.run([self._train_V_op, self._V_mse], feed_dict={
            self._states_feed: states,
            self._V_target_feed: target_V,
            self._is_training_critic: True
        })

    def train_A(self, states, target_A, actions=None):
        use_actions_feed = actions is not None
        if actions is None:
            actions = self.DUMMY_ACTION
        return self.session.run([self._train_A_op, self._A_mse], feed_dict={
            self._states_feed: states,
            self._use_actions_feed: use_actions_feed,
            self._actions_feed: actions,
            self._A_target_feed: target_A,
            self._is_training_a: False,
            self._is_training_critic: True
        })

    def train_Q(self, states, target_Q, actions=None):
        return self.train_A(states, target_Q, actions=actions)

    def train_a(self, states):
        return self.session.run([self._train_a_op, self._av_A, self._infeasibility], feed_dict={
            self._states_feed: states,
            self._use_actions_feed: False,
            self._actions_feed: self.DUMMY_ACTION,
            self._is_training_a: True,
            self._is_training_critic: False
        })

    def train_a_supervised(self, states, desired_actions):
        return self.session.run([self._train_a_supervised_op, self._a_mse], feed_dict={
            self._states_feed: states,
            self._a_desired: desired_actions,
            self._use_actions_feed: False,
            self._actions_feed: self.DUMMY_ACTION,
            self._is_training_a: True,
            self._is_training_critic: False
        })

    def save(self, save_path):
        params = self.session.run(self._get_tf_variables())
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(params, save_path)

    def load(self, load_path):
        params = joblib.load(load_path)
        feed_dict = {}
        for p, p_placeholder in zip(params, self._load_placeholders):
            feed_dict[p_placeholder] = p
        self.session.run(self._load_ops, feed_dict=feed_dict)


class DDPG_Model_Target(DDPG_Model_Base):
    def __init__(self, session: tf.Session, name, main_network: DDPG_Model_Main, env: gym.Env, ob_space, ac_space, constraints,
                 softmax_actor, soft_constraints, cp_optnet, nn_size, init_scale, advantage_learning,
                 use_layer_norm, use_batch_norm, use_norm_actor, tau,
                 log_norm_obs_alloc, log_norm_action, rms_norm_action, **kwargs):
        super().__init__(session=session, name=name, env=env, ob_space=ob_space, ac_space=ac_space, constraints=constraints,
                         softmax_actor=softmax_actor, soft_constraints=soft_constraints, cp_optnet=cp_optnet, nn_size=nn_size,
                         init_scale=init_scale, advantage_learning=advantage_learning,
                         use_layer_norm=use_layer_norm, use_batch_norm=use_batch_norm, use_norm_actor=use_norm_actor,
                         log_norm_obs_alloc=log_norm_obs_alloc, log_norm_action=log_norm_action, rms_norm_action=rms_norm_action)
        logger.log("Setting up target network")
        self.main_network = main_network
        self.tau = tau
        with tf.variable_scope(name):
            self._setup_states_feed()
            self._setup_running_ob_stats()
            self._setup_running_ac_stats()
            self._setup_actor()
            self._setup_critic()
            self._setup_update_from_main_network()

    def _setup_update_from_main_network(self):
        with tf.variable_scope('target_network_update_ops'):
            # for updating target network:
            from_vars = self.main_network._get_tf_variables()
            from_vars_trainable = self.main_network._get_tf_trainable_variables()
            to_vars = self._get_tf_variables()
            to_vars_trainable = self.main_network._get_tf_trainable_variables()
            assert len(from_vars) == len(to_vars) and len(from_vars) > 0, print(
                '{0},{1}'.format(len(from_vars), len(to_vars)))
            assert len(from_vars_trainable) == len(to_vars_trainable) and len(from_vars_trainable) > 0, print(
                '{0},{1}'.format(len(from_vars_trainable), len(to_vars_trainable)))
            self._update_network_op, self._soft_update_network_op = [], []
            for from_var, to_var in zip(from_vars, to_vars):
                hard_update_op = to_var.assign(from_var)
                soft_update_op = to_var.assign(
                    self.tau * from_var + (1 - self.tau) * to_var)
                self._update_network_op.append(hard_update_op)
                # if from_var not in from_vars_trainable:
                #     soft_update_op = hard_update_op
                if 'running_ob_stats' in from_var.name or 'normalized_states' in from_var.name:
                    soft_update_op = hard_update_op
                    logger.log(
                        'Variable {0} will be hard updated to target network'.format(from_var.name))
                self._soft_update_network_op.append(soft_update_op)

    def soft_update_from_main_network(self):
        self.session.run(self._soft_update_network_op)

    def update_from_main_network(self):
        self.session.run(self._update_network_op)


class DDPG_Model_With_Param_Noise(DDPG_Model_Base):
    def __init__(self, session: tf.Session, name, main_network: DDPG_Model_Main, env: gym.Env, target_divergence, ob_space, ac_space, constraints,
                 softmax_actor, soft_constraints, cp_optnet, nn_size, init_scale, advantage_learning,
                 use_layer_norm, use_batch_norm, use_norm_actor,
                 log_norm_obs_alloc, log_norm_action, rms_norm_action, **kwargs):
        super().__init__(session=session, name=name, env=env, ob_space=ob_space, ac_space=ac_space, constraints=constraints,
                         softmax_actor=softmax_actor, soft_constraints=soft_constraints, cp_optnet=cp_optnet, nn_size=nn_size,
                         init_scale=init_scale, advantage_learning=advantage_learning,
                         use_layer_norm=use_layer_norm, use_batch_norm=use_batch_norm, use_norm_actor=use_norm_actor,
                         log_norm_obs_alloc=log_norm_obs_alloc, log_norm_action=log_norm_action, rms_norm_action=rms_norm_action)
        logger.log("Setting up param noise network")
        self.main_network = main_network
        self.target_divergence = target_divergence
        self._main_network_params = self.main_network._get_tf_variables()
        self._main_network_actor_params = list(
            filter(lambda p: 'critic' not in p.name, self._main_network_params))
        self._main_network_params_perturbable = self.main_network._get_tf_perturbable_variables()
        self._main_network_actor_params_perturbable = list(
            filter(lambda p: 'critic' not in p.name, self._main_network_params_perturbable))
        with tf.variable_scope(name):
            self._setup_states_feed()
            self._setup_running_ob_stats()
            self._setup_running_ac_stats()
            self._setup_actor()
            # dont setup critic for this one
            self._setup_update_from_main_network()
            self._setup_divergence_calculation()
            self._setup_param_sensitivity_calculation()
        self.adaptive_sigma = init_scale

    def _setup_update_from_main_network(self):
        with tf.variable_scope('noisy_actor_update_ops'):
            from_vars = self._main_network_actor_params
            to_vars = self._get_tf_variables()
            to_vars_perturbable = self._get_tf_perturbable_variables()
            assert len(from_vars) == len(to_vars) and len(from_vars) > 0, print(
                '{0},{1}'.format(len(from_vars), len(to_vars)))
            assert len(self._main_network_actor_params_perturbable) == len(to_vars_perturbable) and len(to_vars_perturbable) > 0, print(
                '{0},{1}'.format(len(self._main_network_actor_params_perturbable), len(to_vars_perturbable)))
            self._noise_vars = []
            self._noisy_update_network_op = []
            for from_var, to_var in zip(from_vars, to_vars):
                if from_var in self._main_network_actor_params_perturbable:
                    noise_var = tf.placeholder(
                        shape=from_var.shape.as_list(), dtype=tf.float32)
                    self._noise_vars.append(noise_var)
                    self._noisy_update_network_op.append(
                        to_var.assign(from_var + noise_var))
                else:
                    self._noisy_update_network_op.append(
                        to_var.assign(from_var))

    def _setup_param_sensitivity_calculation(self):
        with tf.variable_scope('actor_params_sensitivities'):
            sensitivities_squared = [
                0] * len(self._main_network_actor_params_perturbable)
            for k in range(self.ac_shape[0]):
                gradients_k = tf.gradients(
                    self.main_network._a[:, k], self._main_network_actor_params_perturbable)
                for var_index in range(len(self._main_network_actor_params_perturbable)):
                    sensitivities_squared[var_index] = sensitivities_squared[var_index] + tf.square(
                        gradients_k[var_index])
            self._main_network_actor_params_sensitivities = [
                tf.sqrt(s) for s in sensitivities_squared]

    def _setup_divergence_calculation(self):
        with tf.variable_scope('divergence'):
            self._divergence = tf.sqrt(tf.reduce_mean(
                tf.square(self._a - self.main_network._a)))

    def noisy_update_from_main_network(self, param_noise):
        feed_dict = {}
        for noise_var_placeholder, noise_var in zip(self._noise_vars, param_noise):
            feed_dict[noise_var_placeholder] = noise_var
        return self.session.run(self._noisy_update_network_op, feed_dict=feed_dict)

    def get_divergence(self, states):
        return self.session.run(self._divergence, feed_dict={
            self._states_feed: states,
            self._use_actions_feed: False,
            self._actions_feed: self.DUMMY_ACTION,
            self._is_training_a: False,
            self.main_network._states_feed: states,
            self.main_network._use_actions_feed: False,
            self.main_network._actions_feed: self.DUMMY_ACTION,
            self.main_network._is_training_a: False
        })

    def generate_normal_param_noise(self, sigma=None):
        if sigma is None:
            sigma = self.adaptive_sigma
        params = self.session.run(self._main_network_actor_params_perturbable)
        noise = []
        for p in params:
            n = sigma * np.random.standard_normal(size=np.shape(p))
            noise.append(n)
        return noise

    def generate_safe_noise(self, states, sigma=None):
        noise = self.generate_normal_param_noise(sigma)
        feed_dict = {
            self.main_network._states_feed: states,
            self.main_network._is_training_a: False,
            self.main_network._use_actions_feed: False,
            self.main_network._actions_feed: self.DUMMY_ACTION
        }
        sensitivities = self.session.run(
            self._main_network_actor_params_sensitivities, feed_dict=feed_dict)
        noise_safe = []
        for n, s in zip(noise, sensitivities):
            s = s / np.sqrt(len(states))  # to make s independent of mb size
            n_safe = n / (s + 1e-3)
            noise_safe.append(n_safe)
        return noise_safe

    def adapt_sigma(self, divergence):
        multiplier = 1.15 if max(divergence, self.target_divergence) / \
            (min(divergence, self.target_divergence) + 1e-6) >= 10 else 1.05
        if divergence < self.target_divergence:
            self.adaptive_sigma = self.adaptive_sigma * multiplier
        else:
            self.adaptive_sigma = self.adaptive_sigma / multiplier


class Summaries:
    def __init__(self, session: tf.Session):
        logger.log("Setting up summaries")
        self.session = session
        self.writer = tf.summary.FileWriter(
            logger.get_dir(), self.session.graph)

    def setup_scalar_summaries(self, keys):
        for k in keys:
            # ensure no white spaces in k:
            if ' ' in k:
                raise ValueError("Keys cannot contain whitespaces")
            placeholder_symbol = k
            setattr(self, placeholder_symbol, tf.placeholder(
                dtype=tf.float32, name=placeholder_symbol + '_placeholder'))
            placeholder = getattr(self, placeholder_symbol)
            summay_symbol = k + '_summary'
            setattr(self, summay_symbol, tf.summary.scalar(k, placeholder))

    def setup_histogram_summaries(self, keys):
        for k in keys:
            # ensure no white spaces in k:
            if ' ' in k:
                raise ValueError("Keys cannot contain whitespaces")
            placeholder_symbol = k
            setattr(self, placeholder_symbol, tf.placeholder(
                dtype=tf.float32, shape=[None], name=placeholder_symbol + '_placeholder'))
            placeholder = getattr(self, placeholder_symbol)
            summay_symbol = k + '_summary'
            setattr(self, summay_symbol, tf.summary.histogram(k, placeholder))

    def write_summaries(self, kvs, global_step):
        for key in kvs:
            placeholder_symbol = key
            summary_symbol = key + "_summary"
            if hasattr(self, placeholder_symbol) and hasattr(self, summary_symbol):
                summary = self.session.run(getattr(self, summary_symbol), feed_dict={
                    getattr(self, placeholder_symbol): kvs[key]
                })
                self.writer.add_summary(summary, global_step=global_step)
            else:
                logger.log("Invalid summary key {0}".format(
                    key), level=logger.WARN)


class DDPG_Model:
    def __init__(self, session: tf.Session, use_param_noise, sys_args_dict):
        logger.log("Setting up DDPG Model")
        self.main = DDPG_Model_Main(session, "model_main", **sys_args_dict)
        self.target = DDPG_Model_Target(
            session, "model_target", self.main, **sys_args_dict)
        if use_param_noise:
            self.noisy = DDPG_Model_With_Param_Noise(
                session, "model_param_noise", self.main, **sys_args_dict)
        self.summaries = Summaries(session)
        logger.log("DDPG model setup done")
