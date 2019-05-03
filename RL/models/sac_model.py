import tensorflow as tf
import numpy as np
import RL
from RL.common.utils import tf_inputs, TfRunningStats, auto_conv_dense_net, dense_net, tf_training_step, tf_scale, tf_clip, need_conv_net
import gym


class SACModel:
    def __init__(self, context: RL.Context, name: str, num_critics=1, reuse=tf.AUTO_REUSE):
        self.name = name
        self.context = context
        self.state_space = self.context.env.observation_space
        self.action_space = self.context.env.action_space
        self.num_critics = num_critics
        self.check_assertions()
        with tf.variable_scope(name, reuse=reuse):
            # running stats:
            if self.context.normalize_observations:
                self._states_running_stats = TfRunningStats(list(self.state_space.shape), "states_running_stats")
            if self.context.normalize_actions:
                self._actions_running_stats = TfRunningStats(list(self.action_space.shape), "actions_running_stats")
            # placeholders:
            self._states_placeholder, states_input = tf_inputs([None] + list(self.state_space.shape), self.state_space.dtype, "states", cast_to_dtype=tf.float32)
            self._actions_placeholder, actions_input = tf_inputs([None] + list(self.action_space.shape), self.action_space.dtype, "actions_input", cast_to_dtype=tf.float32)
            self._actions_noise_placholder, actions_noise_input = tf_inputs([None] + list(self.action_space.shape), tf.float32, "actions_noise_input")
            self._actor_loss_coeffs_placeholder, actor_loss_coeffs_input = tf_inputs([self.num_critics], tf.float32, "actor_loss_coefficients")
            self._actor_loss_alpha_placholder, actor_loss_alpha_input = tf_inputs(None, tf.float32, "actor_loss_alpha_coefficient")
            self._critics_loss_coeffs_placholder, critics_loss_coeffs_input = tf_inputs([self.num_critics], tf.float32, "critics_loss_coefficients")
            self._critics_targets_placeholder, critics_targets_input = tf_inputs([self.num_critics, None], tf.float32, "critics_targets")
            # normalized inputs:
            states_input_normalized = self._states_running_stats.normalize(states_input, "states_input_normalize") if self.context.normalize_observations else states_input
            actions_input_normalized = self._actions_running_stats.normalize(actions_input, "actions_input_normalize") if self.context.normalize_actions else actions_input
            # actor:
            self._actor_actions, self._actor_means, self._actor_logstds, self._actor_logpis = self.tf_actor(states_input_normalized, actions_noise_input, 'actor')
            # critics:
            self._critics = []
            for i in range(self.num_critics):
                critic = self.tf_critic(states_input_normalized, actions_input_normalized, "critic{0}".format(i))
                self._critics.append(critic)
            # actor-critics:
            actor_actions_normalized = self._actions_running_stats.normalize(self._actor_actions, "actor_actions_normalize") if self.context.normalize_actions else self._actor_actions
            self._actor_critics = []
            for i in range(self.num_critics):
                actor_critic = self.tf_critic(states_input_normalized, actor_actions_normalized, "critic{0}".format(i), reuse=True)
                self._actor_critics.append(actor_critic)
            # actor-loss:
            self._actor_loss = self.tf_actor_loss(actor_loss_coeffs_input, actor_loss_alpha_input, self._actor_critics, self._actor_logpis, "actor_loss")
            # critic-loss:
            self._critics_loss = self.tf_critic_loss(critics_loss_coeffs_input, self._critics, critics_targets_input, "critics_loss")
        self.params = self.get_vars()

    def check_assertions(self):
        if not hasattr(self.state_space, 'dtype'):
            self.state_space.dtype = np.float32
        if not hasattr(self.action_space, 'dtype'):
            self.action_space.dtype = np.float32
        assert isinstance(self.state_space, gym.spaces.Box)
        assert isinstance(self.action_space, gym.spaces.Box)
        assert len(self.action_space.shape) == 1
        assert self.num_critics >= 1, "There should be atleast one critic, against which the actor is optimized"

    def tf_actor_activation_fn(self, means, logstds, noise, name):
        with tf.variable_scope(name):
            # gaussian actions
            stds = tf.exp(logstds)
            x = means + noise * stds
            logpi_per_dimension = -tf.square((x - means) / (stds + 1e-8)) - logstds - np.log(np.sqrt(2 * np.pi))  # log of gaussian per dimension
            logpi = tf.reduce_sum(logpi_per_dimension, axis=-1)  # overall pi is product of pi per dimension, so overall log_pi is sum of log_pi per dimension
            # tanh
            y = tf.nn.tanh(x)
            dy_dx = tf_clip(1 - tf.square(y), 0, 1)
            log_dy_dx = tf.log(dy_dx + 1e-6)
            # scale
            z = tf_scale(y, -1, 1, self.action_space.low, self.action_space.high, "scale")
            dz_dy = (self.action_space.high - self.action_space.low) / 2
            log_dz_dy = tf.log(dz_dy)
            # overall log derivative:
            log_dz_dx = log_dz_dy + log_dy_dx  # by chain rule. it is a sum due to log operator
            # by change of variables: overall log probability after activations = logpi - component wise sum of log derivatives of the activation
            logpi = logpi - tf.reduce_sum(log_dz_dx, axis=-1)
            return z, logpi

    def tf_actor(self, states, actions_noise, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name):
            outp = auto_conv_dense_net(need_conv_net(self.context.env.observation_space), states, self.context.convs, self.context.hidden_layers, self.context.activation_fn, 2 * self.action_space.shape[0], lambda x: x, "conv_dense", output_kernel_initializer=tf.random_uniform_initializer(minval=-self.context.init_scale, maxval=self.context.init_scale), reuse=reuse)
            means = outp[:, 0:self.action_space.shape[0]]
            # logstds = tf.tanh(outp[:, self.action_space.shape[0]:])
            # logstds = tf_scale(logstds, -1, 1, self.context.logstd_min, self.context.logstd_max, 'scale_logstd')
            logstds = tf_clip(outp[:, self.action_space.shape[0]:], self.context.logstd_min, self.context.logstd_max)
            actions, logpi = self.tf_actor_activation_fn(means, logstds, actions_noise, 'gaussian_tanh_scaled')
            return actions, means, logstds, logpi

    def tf_critic(self, states, actions, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            states_to_one_hidden = auto_conv_dense_net(need_conv_net(self.context.env.observation_space), states, self.context.convs, self.context.hidden_layers[:1], self.context.activation_fn, None, None, "conv_dense", reuse=reuse)
            one_hidden_plus_actions = tf.concat(values=[states_to_one_hidden, actions], axis=1, name="concat")
            return dense_net(one_hidden_plus_actions, self.context.hidden_layers[1:], self.context.activation_fn, 1, lambda x: x, "dense", output_kernel_initializer=tf.random_uniform_initializer(minval=-self.context.init_scale, maxval=self.context.init_scale), reuse=reuse)[:, 0]

    def tf_actor_loss(self, actor_loss_coeffs, actor_loss_alpha, actor_critics, actor_logpis, name):
        with tf.variable_scope(name):
            loss = 0
            for i in range(self.num_critics):
                loss -= actor_loss_coeffs[i] * actor_critics[i]
            loss += actor_loss_alpha * actor_logpis
            loss = tf.reduce_mean(loss)
            return loss

    def tf_critic_loss(self, critics_loss_coeffs, critics, critics_targets, name):
        '''assumes the first axis is critic_id'''
        with tf.variable_scope(name):
            loss = 0
            for i in range(self.num_critics):
                loss += critics_loss_coeffs[i] * tf.losses.mean_squared_error(critics[i], critics_targets[i])
            return loss

    def get_vars(self, scope=''):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{0}/{1}'.format(self.name, scope))

    def get_trainable_vars(self, scope=''):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/{1}'.format(self.name, scope))

    def get_perturbable_vars(self, scope=''):
        return list(filter(lambda var: not('LayerNorm' in var.name or 'batch_norm' in var.name or 'running_stats' in var.name), self.get_vars(scope)))

    def sample_actions_noise(self, batch_size, sigma=1):
        if hasattr(sigma, '__len__'):
            sigma = np.asarray(sigma)
            sigma = np.reshape(sigma, [batch_size, 1])
        return sigma * np.random.standard_normal(size=[batch_size] + list(self.action_space.shape))

    def actions(self, states, noise=None):
        actions = self._actor_actions
        if noise is None:
            actions = self._actor_means
            noise = np.zeros([len(states), self.context.env.action_space.shape[0]])
        return self.context.session.run(actions, {
            self._states_placeholder: states,
            self._actions_noise_placholder: noise
        })

    def actions_means_logstds_logpis(self, states, noise=None):
        actions = self._actor_actions
        if noise is None:
            actions = self._actor_means
            noise = np.zeros([len(states), self.context.env.action_space.shape[0]])
        return self.context.session.run([actions, self._actor_means, self._actor_logstds, self._actor_logpis], {
            self._states_placeholder: states,
            self._actions_noise_placholder: noise
        })

    def Q(self, critic_ids, states, actions):
        '''returns the list of state-action values per critic specified in critic_ids'''
        return self.context.session.run([self._critics[i] for i in critic_ids], {self._states_placeholder: states, self._actions_placeholder: actions})

    def setup_training(self, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            # running stats update
            if self.context.normalize_observations:
                self._update_states_running_stats = self._states_running_stats.update(self._states_placeholder[0], "update_states_running_stats")
            if self.context.normalize_actions:
                self._update_actions_running_stats = self._actions_running_stats.update(self._actions_placeholder[0], "update_actions_running_stats")
            # actor training
            actor_trainable_vars = self.get_trainable_vars('actor')
            actor_optimizer = tf.train.AdamOptimizer(self.context.actor_learning_rate)
            assert len(actor_trainable_vars) > 0, "No vars to train in actor"
            self._actor_train_step = tf_training_step(self._actor_loss, actor_trainable_vars, actor_optimizer, self.context.actor_l2_reg, self.context.clip_gradients, "actor_train_step")
            # critics training
            critics_trainable_vars = []
            for i in range(self.num_critics):
                critics_trainable_vars += self.get_trainable_vars('critic{0}'.format(i))
            critics_optimizer = tf.train.AdamOptimizer(self.context.learning_rate)
            assert len(critics_trainable_vars) > 0, "No vars to train in critics"
            self._critics_train_step = tf_training_step(self._critics_loss, critics_trainable_vars, critics_optimizer, self.context.l2_reg, self.context.clip_gradients, "critics_train_step")
            return self._actor_train_step, self._critics_train_step

    def update_states_running_stats(self, states):
        for i in range(len(states)):
            self.context.session.run(self._update_states_running_stats, {self._states_placeholder: [states[i]]})

    def update_actions_running_stats(self, actions):
        for i in range(len(actions)):
            self.context.session.run(self._update_actions_running_stats, {self._actions_placeholder: [actions[i]]})

    def train_actor(self, states, noise, critic_ids, loss_coeffs, alpha):
        '''train the actor to optimize the critics specified by critic_ids weighted by loss_coeffs and optimize entropy weighted by alpha'''
        loss_coeffs_all_ids = [0] * self.num_critics
        actor_critics = []
        for i, coeff in zip(critic_ids, loss_coeffs):
            loss_coeffs_all_ids[i] = coeff
            actor_critics.append(self._actor_critics[i])
        _, loss, actor_critics, logstds, logpis = self.context.session.run([self._actor_train_step, self._actor_loss, actor_critics, self._actor_logstds, self._actor_logpis], {
            self._states_placeholder: states,
            self._actions_noise_placholder: noise,
            self._actor_loss_coeffs_placeholder: loss_coeffs_all_ids,
            self._actor_loss_alpha_placholder: alpha
        })
        return loss, actor_critics, logstds, logpis

    def train_critics(self, states, actions, critic_ids, critics_targets, critics_loss_coeffs):
        '''jointly train the critics of given ids with given loss coeffs. critics_targets is expected to be a list of targets per critic to train'''
        critics_targets_all_ids = np.zeros([self.num_critics, len(states)])
        critics_loss_coeffs_all_ids = [0] * self.num_critics
        for i, targets, loss_coeff in zip(critic_ids, critics_targets, critics_loss_coeffs):
            critics_loss_coeffs_all_ids[i] = loss_coeff
            critics_targets_all_ids[i] = targets
        _, critics_loss = self.context.session.run([self._critics_train_step, self._critics_loss], feed_dict={
            self._states_placeholder: states,
            self._actions_placeholder: actions,
            self._critics_targets_placeholder: critics_targets_all_ids,
            self._critics_loss_coeffs_placholder: critics_loss_coeffs_all_ids
        })
        return critics_loss
