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
            self._states_running_stats = TfRunningStats(list(self.state_space.shape), "states_running_stats")
            self._actions_running_stats = TfRunningStats(list(self.action_space.shape), "actions_running_stats")
            # placeholders:
            self._states_placeholder, states_input = tf_inputs([None] + list(self.state_space.shape), self.state_space.dtype, "states", cast_to_dtype=tf.float32)
            self._actions_placeholder, actions_input = tf_inputs([None] + list(self.action_space.shape), self.action_space.dtype, "actions_input", cast_to_dtype=tf.float32)
            self._actions_noise_placholder, actions_noise_input = tf_inputs([None] + list(self.action_space.shape), tf.float32, "actions_noise_input")
            # normalized inputs:
            states_input_normalized = self._states_running_stats.normalize(states_input, "states_input_normalize")
            actions_input_normalized = self._actions_running_stats.normalize(actions_input, "actions_input_normalize")
            # actor:
            self._actor_actions, self._actor_means, self._actor_logstds, self._actor_logpis = self.tf_actor(states_input_normalized, actions_noise_input, 'actor')
            # critics:
            self._critics_Qs = []
            for i in range(self.num_critics):
                critic_Q = self.tf_critic(states_input_normalized, actions_input_normalized, "critic{0}".format(i))
                self._critics_Qs.append(critic_Q)
            # actor-critic:
            actor_actions_normalized = self._actions_running_stats.normalize(self._actor_actions, "actor_actions_normalize")
            self._actor_critic_Q = self.tf_critic(states_input_normalized, actor_actions_normalized, "critic0", reuse=True)
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

    def tf_actor_activation_fn(self, x, logpi):
        # tanh
        y = tf.nn.tanh(x)
        dy_dx = tf_clip(1 - tf.square(y), 0, 1)
        log_dy_dx = tf.log(dy_dx + 1e-6)
        # scale
        z = tf_scale(y, -1, 1, self.action_space.low, self.action_space.high, "scale")
        dz_dy = (self.action_space.high - self.action_space.low) / 2
        log_dz_dy = tf.log(dz_dy)
        # overall log derivative:
        log_dz_dx = log_dz_dy + log_dy_dx
        # by change of variables: overall log probability after activations = log_pi - component wise sum of log derivatives of the activation
        logpi = logpi - tf.reduce_sum(log_dz_dx, axis=-1)
        return z, logpi

    def tf_actor(self, states, actions_noise, name, reuse=tf.AUTO_REUSE):
        outp = auto_conv_dense_net(need_conv_net(self.context.env.observation_space), states, self.context.convs, self.context.hidden_layers, self.context.activation_fn, 2 * self.action_space.shape[0], lambda x: x, name, output_kernel_initializer=tf.random_uniform_initializer(minval=-self.context.init_scale, maxval=self.context.init_scale), reuse=reuse)
        means = outp[:, 0:self.action_space.shape[0]]
        # logstds = tf.tanh(outp[:, self.action_space.shape[0]:])
        # logstds = tf_scale(logstds, -1, 1, self.context.logstd_min, self.context.logstd_max, 'scale_logstd')
        logstds = tf_clip(outp[:, self.action_space.shape[0]:], self.context.logstd_min, self.context.logstd_max)
        stds = tf.exp(logstds)
        x = means + actions_noise * stds  # gaussian actions
        logpi_per_dimension = -tf.square((x - means) / (stds + 1e-8)) - logstds - np.log(np.sqrt(2 * np.pi))  # log of gaussian per dimension
        logpi = tf.reduce_sum(logpi_per_dimension, axis=-1)  # overall pi is product of pi per dimension, so overall log_pi is sum of log_pi per dimension
        actions, logpi = self.tf_actor_activation_fn(x, logpi)
        return actions, means, logstds, logpi

    def tf_critic(self, states, actions, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            states_to_one_hidden = auto_conv_dense_net(need_conv_net(self.context.env.observation_space), states, self.context.convs, self.context.hidden_layers[:1], self.context.activation_fn, None, None, "one_hidden", reuse=reuse)
            one_hidden_plus_actions = tf.concat(values=[states_to_one_hidden, actions], axis=1, name="concat")
            return dense_net(one_hidden_plus_actions, self.context.hidden_layers[1:], self.context.activation_fn, 1, lambda x: x, "Q", output_kernel_initializer=tf.random_uniform_initializer(minval=-self.context.init_scale, maxval=self.context.init_scale), reuse=reuse)[:, 0]

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

    def Q(self, critic_id, states, actions):
        return self.context.session.run(self._critics_Qs[critic_id], {self._states_placeholder: states, self._actions_placeholder: actions})

    def setup_training(self, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            # running stats update
            self._update_states_running_stats = self._states_running_stats.update(self._states_placeholder[0], "update_states_running_stats")
            self._update_actions_running_stats = self._actions_running_stats.update(self._actions_placeholder[0], "update_actions_running_stats")
            # actor training
            self._alpha, _ = tf_inputs(None, tf.float32, 'alpha')
            self._actor_loss = -tf.reduce_mean(self._actor_critic_Q - self._alpha * self._actor_logpis)
            actor_trainable_vars = self.get_trainable_vars('actor')
            actor_optimizer = tf.train.AdamOptimizer(self.context.actor_learning_rate)
            assert len(actor_trainable_vars) > 0, "No vars to train in actor"
            self._actor_train_step = tf_training_step(self._actor_loss, actor_trainable_vars, actor_optimizer, self.context.actor_l2_reg, self.context.clip_gradients, "actor_train_step")
            # critics training
            self._Q_targets_placholders = []
            self._critics_losses = []
            self._critics_train_steps = []
            for i in range(self.num_critics):
                Q_targets_placholder, _ = tf_inputs([None], tf.float32, "Q_targets{0}".format(i))
                critic_loss = tf.losses.mean_squared_error(self._critics_Qs[i], Q_targets_placholder)
                critic_trainable_vars = self.get_trainable_vars('critic{0}'.format(i))
                critic_optimizer = tf.train.AdamOptimizer(self.context.learning_rate)
                assert len(critic_trainable_vars) > 0, "No vars to train in critic{0}".format(i)
                critic_train_step = tf_training_step(critic_loss, critic_trainable_vars, critic_optimizer, self.context.l2_reg, self.context.clip_gradients, "critic{0}_train_step".format(i))
                self._Q_targets_placholders.append(Q_targets_placholder)
                self._critics_losses.append(critic_loss)
                self._critics_train_steps.append(critic_train_step)
            return self._actor_train_step, self._critics_train_steps

    def update_states_running_stats(self, states):
        for i in range(len(states)):
            self.context.session.run(self._update_states_running_stats, {self._states_placeholder: [states[i]]})

    def update_actions_running_stats(self, actions):
        for i in range(len(actions)):
            self.context.session.run(self._update_actions_running_stats, {self._actions_placeholder: [actions[i]]})

    def train_actor(self, states, noise, alpha):
        _, loss, Q, logstds, logpis = self.context.session.run([self._actor_train_step, self._actor_loss, self._actor_critic_Q, self._actor_logstds, self._actor_logpis], {
            self._states_placeholder: states,
            self._actions_noise_placholder: noise,
            self._alpha: alpha
        })
        return loss, Q, logstds, logpis

    def train_critic(self, critic_id, states, actions, Q_targets):
        _, critic_loss = self.context.session.run([self._critics_train_steps[critic_id], self._critics_losses[critic_id]], feed_dict={
            self._states_placeholder: states,
            self._actions_placeholder: actions,
            self._Q_targets_placholders[critic_id]: Q_targets
        })
        return critic_loss
