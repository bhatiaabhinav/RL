import tensorflow as tf
import numpy as np
import RL
from RL.common.utils import tf_inputs, TfRunningStats, conv_net, auto_conv_dense_net, dense_net, tf_training_step, tf_scale, tf_clip, need_conv_net


class SACModel:
    def __init__(self, context: RL.Context, name: str, num_actors=1, num_critics=1, num_valuefns=1, reuse=tf.AUTO_REUSE):
        self.name = name
        self.context = context
        self.state_space = self.context.env.observation_space
        self.action_space = self.context.env.action_space
        self.num_actors = num_actors
        self.num_critics = num_critics
        self.num_valuefns = num_valuefns
        self.check_assertions()
        with tf.variable_scope(name, reuse=reuse):
            # running stats:
            if self.context.normalize_observations:
                self._states_running_stats = TfRunningStats(list(self.state_space.shape), "running_stats/states")
            if self.context.normalize_actions:
                self._actions_running_stats = TfRunningStats(list(self.action_space.shape), "running_stats/actions")
            # placeholders:
            self._states_placeholder, states_input = tf_inputs([None] + list(self.state_space.shape), self.state_space.dtype, "states", cast_to_dtype=tf.float32)
            self._actions_placeholder, actions_input = tf_inputs([None] + list(self.action_space.shape), self.action_space.dtype, "actions_input", cast_to_dtype=tf.float32)
            self._actions_noise_placholder, actions_noise_input = tf_inputs([None] + list(self.action_space.shape), tf.float32, "actions_noise_input")
            self._actor_loss_coeffs_placeholder, actor_loss_coeffs_input = tf_inputs([self.num_critics], tf.float32, "actor_loss_coefficients")
            self._actor_loss_alpha_placholder, actor_loss_alpha_input = tf_inputs(None, tf.float32, "actor_loss_alpha_coefficient")
            self._critics_loss_coeffs_placeholder, critics_loss_coeffs_input = tf_inputs([self.num_critics], tf.float32, "critics_loss_coefficients")
            self._critics_targets_placeholder, critics_targets_input = tf_inputs([self.num_critics, None], tf.float32, "critics_targets")
            self._valuefns_loss_coeffs_placeholder, valuefns_loss_coeffs_input = tf_inputs([self.num_valuefns], tf.float32, "valuefns_loss_coefficients")
            self._valuefns_targets_placeholder, valuefns_targets_input = tf_inputs([self.num_valuefns, None], tf.float32, "valuefns_targets")
            # normalized inputs:
            states_input_normalized = self._states_running_stats.normalize(states_input, "states_input_normalize") if self.context.normalize_observations else states_input
            actions_input_normalized = self._actions_running_stats.normalize(actions_input, "actions_input_normalize") if self.context.normalize_actions else actions_input
            # critics:
            self._critics = [self.tf_critic(states_input_normalized, actions_input_normalized, "critic{0}".format(i)) for i in range(self.num_critics)]
            self._critics_loss = self.tf_critics_loss(critics_loss_coeffs_input, self._critics, critics_targets_input, "critics_loss")
            # value functions:
            self._valuefns = [self.tf_value_fn(states_input_normalized, "valuefn{0}".format(i)) for i in range(self.num_valuefns)]
            self._valuefns_loss = self.tf_valuefns_loss(valuefns_loss_coeffs_input, self._valuefns, valuefns_targets_input, "valuefns_loss")
            if self.num_actors:
                # actor
                self._actor_means, self._actor_logstds, self._actor_actions, self._actor_logpis = self.tf_actor(states_input_normalized, actions_noise_input, 'actor')
                # actor normalized
                actor_actions_normalized = self._actions_running_stats.normalize(self._actor_actions, "actor_actions_normalize") if self.context.normalize_actions else self._actor_actions
                # actor-critics
                self._actor_critics = [self.tf_critic(states_input_normalized, actor_actions_normalized, "critic{0}".format(i), reuse=True) for i in range(self.num_critics)]
                # actor-loss:
                self._actor_loss = self.tf_actor_loss(actor_loss_coeffs_input, actor_loss_alpha_input, self._actor_critics, self._actor_logpis, "actor_loss")

    def check_assertions(self):
        if not hasattr(self.state_space, 'dtype'):
            self.state_space.dtype = np.float32
        if not hasattr(self.action_space, 'dtype'):
            self.action_space.dtype = np.float32
        assert hasattr(self.state_space, 'shape')
        assert hasattr(self.action_space, 'shape') and hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high')
        assert len(self.action_space.shape) == 1
        assert self.num_actors <= 1, "There can be atmost 1 actor"

    def tf_actor_activation_fn(self, means, logstds, noise, name):
        with tf.variable_scope(name):
            # gaussian actions
            stds = tf.exp(logstds)
            x = means + noise * stds
            logpi_per_dimension = -tf.square((x - means) / (stds + 1e-8)) - logstds - np.log(np.sqrt(2 * np.pi))  # log of gaussian per dimension
            logpi = tf.reduce_sum(logpi_per_dimension, axis=-1)  # overall pi is product of pi per dimension, so overall log_pi is sum of log_pi per dimension
            # tanh
            means = tf.nn.tanh(means)
            y = tf.nn.tanh(x)
            dy_dx = tf_clip(1 - tf.square(y), 0, 1)
            log_dy_dx = tf.log(dy_dx + 1e-6)
            logpi = logpi - tf.reduce_sum(log_dy_dx, axis=-1)  # by change of variables: overall log probability after activations = logpi - component wise sum of log derivatives of the activation
            return means, y, logpi

    def tf_actions_scale(self, means, actions, name):
        '''Does not change logpi'''
        with tf.variable_scope(name):
            actions = tf_scale(actions, -1, 1, self.action_space.low, self.action_space.high, "scale_actions")
            means = tf_scale(means, -1, 1, self.action_space.low, self.action_space.high, "scale_mean")
            # dy_dx = ((self.action_space.high - self.action_space.low) / 2).astype(np.float32)
            # log_dy_dx = tf.log(dy_dx)
            # overall log derivative:
            # by change of variables: overall log probability after activations = logpi - component wise sum of log derivatives of the activation
            # logpi = logpi - tf.reduce_sum(log_dy_dx, axis=-1)
            return means, actions

    def tf_actor(self, states, actions_noise, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            outp = auto_conv_dense_net(need_conv_net(self.context.env.observation_space), states, self.context.convs, self.context.hidden_layers, self.context.activation_fn, 2 * self.action_space.shape[0], lambda x: x, "conv_dense", output_kernel_initializer=tf.random_uniform_initializer(minval=-self.context.init_scale, maxval=self.context.init_scale), reuse=reuse)
            means = outp[:, 0:self.action_space.shape[0]]
            # logstds = tf.tanh(outp[:, self.action_space.shape[0]:])
            # logstds = tf_scale(logstds, -1, 1, self.context.logstd_min, self.context.logstd_max, 'scale_logstd')
            logstds = tf_clip(outp[:, self.action_space.shape[0]:], self.context.logstd_min, self.context.logstd_max)
            means, actions, logpi = self.tf_actor_activation_fn(means, logstds, actions_noise, 'gaussian_tanh')
            means, actions = self.tf_actions_scale(means, actions, "scale")
            return means, logstds, actions, logpi

    def tf_critic(self, states, actions, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            if need_conv_net(self.context.env.observation_space):
                states = conv_net(states, self.context.convs, self.context.activation_fn, 'conv', reuse=reuse)
            states_actions = tf.concat(values=[states, actions], axis=-1)
            return dense_net(states_actions, self.context.hidden_layers, self.context.activation_fn, 1, lambda x: x, "dense", output_kernel_initializer=tf.random_uniform_initializer(minval=-self.context.init_scale, maxval=self.context.init_scale), reuse=reuse)[:, 0]

    def tf_value_fn(self, states, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            return auto_conv_dense_net(need_conv_net(self.context.env.observation_space), states, self.context.convs, self.context.hidden_layers, self.context.activation_fn, 1, lambda x: x, "conv_dense", output_kernel_initializer=tf.random_uniform_initializer(minval=-self.context.init_scale, maxval=self.context.init_scale), reuse=reuse)[:, 0]

    def tf_actor_loss(self, actor_loss_coeffs, actor_loss_alpha, actor_critics, actor_logpis, name):
        with tf.variable_scope(name):
            loss = 0
            loss = sum([-actor_loss_coeffs[i] * actor_critics[i] for i in range(self.num_critics)]) + actor_loss_alpha * actor_logpis
            loss = tf.reduce_mean(loss)
            return loss

    def tf_critics_loss(self, critics_loss_coeffs, critics, critics_targets, name):
        '''assumes the first axis is critic_id'''
        with tf.variable_scope(name):
            loss = sum([critics_loss_coeffs[i] * tf.losses.mean_squared_error(critics[i], critics_targets[i]) for i in range(self.num_critics)])
            return loss

    def tf_valuefns_loss(self, valuefns_loss_coeffs, valuefns, valuefns_targets, name):
        '''assumes the first axis is valuefn_id'''
        with tf.variable_scope(name):
            loss = sum([valuefns_loss_coeffs[i] * tf.losses.mean_squared_error(valuefns[i], valuefns_targets[i]) for i in range(self.num_valuefns)])
            return loss

    def get_vars(self, *scopes):
        if len(scopes) == 0:
            scopes = ['']
        return sum([tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{0}/{1}'.format(self.name, scope)) for scope in scopes], [])

    def get_trainable_vars(self, *scopes):
        if len(scopes) == 0:
            scopes = ['']
        return sum([tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/{1}'.format(self.name, scope)) for scope in scopes], [])

    def get_perturbable_vars(self, *scopes):
        return list(filter(lambda var: not('LayerNorm' in var.name or 'batch_norm' in var.name or 'running_stats' in var.name), self.get_vars(*scopes)))

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

    def V(self, valuefn_ids, states):
        '''returns the list of state values per valuefn specified in valuefn_ids'''
        return self.context.session.run([self._valuefns[i] for i in valuefn_ids], {self._states_placeholder: states})

    def setup_training(self, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            # running stats update
            if self.context.normalize_observations:
                self._update_states_running_stats = self._states_running_stats.update(self._states_placeholder[0], "update_states_running_stats")
            if self.context.normalize_actions:
                self._update_actions_running_stats = self._actions_running_stats.update(self._actions_placeholder[0], "update_actions_running_stats")
            # actor training
            if self.num_actors:
                actor_trainable_vars = self.get_trainable_vars('actor')
                actor_optimizer = tf.train.AdamOptimizer(self.context.actor_learning_rate)
                assert len(actor_trainable_vars) > 0, "No vars to train in actor"
                self._actor_train_step = tf_training_step(self._actor_loss, actor_trainable_vars, actor_optimizer, self.context.actor_l2_reg, self.context.clip_gradients, "actor_train_step")
            # critics training
            if self.num_critics:
                critics_trainable_vars = self.get_trainable_vars(*('critic{0}'.format(i) for i in range(self.num_critics)))
                critics_optimizer = tf.train.AdamOptimizer(self.context.learning_rate)
                assert len(critics_trainable_vars) > 0, "No vars to train in critics"
                self._critics_train_step = tf_training_step(self._critics_loss, critics_trainable_vars, critics_optimizer, self.context.l2_reg, self.context.clip_gradients, "critics_train_step")
            # valuefns training
            if self.num_valuefns:
                valuefns_trainable_vars = self.get_trainable_vars(*('valuefn{0}'.format(i) for i in range(self.num_valuefns)))
                valuefns_optimizer = tf.train.AdamOptimizer(self.context.learning_rate)
                assert len(valuefns_trainable_vars) > 0, "No vars to train in valuefns"
                self._valuefns_train_step = tf_training_step(self._valuefns_loss, valuefns_trainable_vars, valuefns_optimizer, self.context.l2_reg, self.context.clip_gradients, "valuefns_train_step")

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

    def train_critics(self, states, actions, critic_ids, targets, loss_coeffs):
        '''jointly train the critics of given ids with given loss coeffs. targets is expected to be a list of targets per critic to train. i.e. of shape [len(critic_ids), len(states)]'''
        targets_all_ids = np.zeros([self.num_critics, len(states)])
        loss_coeffs_all_ids = [0] * self.num_critics
        for i, tar, coeff in zip(critic_ids, targets, loss_coeffs):
            loss_coeffs_all_ids[i] = coeff
            targets_all_ids[i] = tar
        _, critics_loss = self.context.session.run([self._critics_train_step, self._critics_loss], feed_dict={
            self._states_placeholder: states,
            self._actions_placeholder: actions,
            self._critics_targets_placeholder: targets_all_ids,
            self._critics_loss_coeffs_placeholder: loss_coeffs_all_ids
        })
        return critics_loss

    def train_valuefns(self, states, valuefn_ids, targets, loss_coeffs):
        '''jointly train the valuefns of given ids with given loss coeffs. targets is expected to be a list of targets per valuefn to train i.e. of shape [len(valuefn_ids), len(states)]'''
        targets_all_ids = np.zeros([self.num_valuefns, len(states)])
        loss_coeffs_all_ids = [0] * self.num_valuefns
        for i, tar, coeff in zip(valuefn_ids, targets, loss_coeffs):
            loss_coeffs_all_ids[i] = coeff
            targets_all_ids[i] = tar
        _, valuefns_loss = self.context.session.run([self._valuefns_train_step, self._valuefns_loss], feed_dict={
            self._states_placeholder: states,
            self._valuefns_targets_placeholder: targets_all_ids,
            self._valuefns_loss_coeffs_placeholder: loss_coeffs_all_ids
        })
        return valuefns_loss
