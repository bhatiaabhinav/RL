import tensorflow as tf
import numpy as np
import RL
from RL.common.utils import tf_inputs, TfRunningStats, auto_conv_dense_net, dense_net, tf_training_step, tf_scale, need_conv_net
import gym


class DDPGModel:
    def __init__(self, context: RL.Context, name: str, reuse=tf.AUTO_REUSE):
        self.name = name
        self.context = context
        self.state_space = self.context.env.observation_space
        self.action_space = self.context.env.action_space
        self.check_assertions()
        with tf.variable_scope(name, reuse=reuse):
            # running stats:
            self._states_running_stats = TfRunningStats(list(self.state_space.shape), "states_running_stats")
            self._actions_running_stats = TfRunningStats(list(self.action_space.shape), "actions_running_stats")
            # placeholders:
            self._states_placeholder, states_input = tf_inputs([None] + list(self.state_space.shape), self.state_space.dtype, "states", cast_to_dtype=tf.float32)
            self._actions_placeholder, actions_input = tf_inputs([None] + list(self.action_space.shape), self.action_space.dtype, "actions_input", cast_to_dtype=tf.float32)
            # normalized inputs:
            states_input_normalized = self._states_running_stats.normalize(states_input, "states_input_normalize")
            actions_input_normalized = self._actions_running_stats.normalize(actions_input, "actions_input_normalize")
            # actor:
            self._actor_actions = self.tf_actor(states_input_normalized, 'actor')
            # critic:
            self._critic_Q = self.tf_critic(states_input_normalized, actions_input_normalized, "critic")
            # actor-critic:
            actor_actions_normalized = self._actions_running_stats.normalize(self._actor_actions, "actor_actions_normalize")
            self._actor_critic_Q = self.tf_critic(states_input_normalized, actor_actions_normalized, "critic", reuse=True)
        self.params = self.get_vars()

    def check_assertions(self):
        if not hasattr(self.state_space, 'dtype'):
            self.state_space.dtype = np.float32
        if not hasattr(self.action_space, 'dtype'):
            self.action_space.dtype = np.float32
        assert isinstance(self.state_space, gym.spaces.Box)
        assert isinstance(self.action_space, gym.spaces.Box)
        assert len(self.action_space.shape) == 1

    def tf_actor_activation_fn(self, x):
        y = tf.nn.tanh(x)
        y = tf_scale(y, -1, 1, self.action_space.low, self.action_space.high, "scale")
        return y

    def tf_actor(self, states, name, reuse=tf.AUTO_REUSE):
        return auto_conv_dense_net(need_conv_net(self.context.env.observation_space), states, self.context.convs, self.context.hidden_layers, self.context.activation_fn, self.action_space.shape[0], self.tf_actor_activation_fn, name, output_kernel_initializer=tf.random_uniform_initializer(minval=-self.context.init_scale, maxval=self.context.init_scale), reuse=reuse)

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

    def actions(self, states):
        return self.context.session.run(self._actor_actions, {self._states_placeholder: states})

    def Q(self, states, actions=None):
        if actions is None:
            return self.context.session.run(self._actor_critic_Q, {self._states_placeholder: states})
        else:
            return self.context.session.run(self._critic_Q, {self._states_placeholder: states, self._actions_placeholder: actions})

    def setup_training(self, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            # running stats update
            self._update_states_running_stats = self._states_running_stats.update(self._states_placeholder[0], "update_states_running_stats")
            self._update_actions_running_stats = self._actions_running_stats.update(self._actions_placeholder[0], "update_actions_running_stats")
            # actor training
            self._actor_loss = -tf.reduce_mean(self._actor_critic_Q)
            actor_trainable_vars = self.get_trainable_vars('actor')
            actor_optimizer = tf.train.AdamOptimizer(self.context.actor_learning_rate)
            assert len(actor_trainable_vars) > 0, "No vars to train!"
            self._actor_train_step = tf_training_step(self._actor_loss, actor_trainable_vars, actor_optimizer, self.context.actor_l2_reg, self.context.clip_gradients, "actor_train_step")
            # critic training
            self._Q_targets_placholder, _ = tf_inputs([None], tf.float32, "Q_targets")
            self._critic_loss = tf.losses.huber_loss(self._critic_Q, self._Q_targets_placholder)
            critic_trainable_vars = self.get_trainable_vars('critic')
            critic_optimizer = tf.train.AdamOptimizer(self.context.learning_rate)
            assert len(critic_trainable_vars) > 0, "No vars to train!"
            self._critic_train_step = tf_training_step(self._critic_loss, critic_trainable_vars, critic_optimizer, self.context.l2_reg, self.context.clip_gradients, "critic_train_step")
            return self._actor_train_step, self._critic_train_step

    def update_states_running_stats(self, states):
        for i in range(len(states)):
            self.context.session.run(self._update_states_running_stats, {self._states_placeholder: [states[i]]})

    def update_actions_running_stats(self, actions):
        for i in range(len(actions)):
            self.context.session.run(self._update_actions_running_stats, {self._actions_placeholder: [actions[i]]})

    def train_actor(self, states):
        _, actor_loss = self.context.session.run([self._actor_train_step, self._actor_loss], {self._states_placeholder: states})
        return actor_loss

    def train_critic(self, states, actions, Q_targets):
        _, critic_loss = self.context.session.run([self._critic_train_step, self._critic_loss], feed_dict={
            self._states_placeholder: states,
            self._actions_placeholder: actions,
            self._Q_targets_placholder: Q_targets
        })
        return critic_loss
