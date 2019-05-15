import tensorflow as tf
import numpy as np
import RL
from RL.common.utils import tf_inputs, TfRunningStats, auto_conv_dense_net, dense_net, tf_training_step, tf_scale, need_conv_net


class DDPGModel:
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
                self._states_running_stats = TfRunningStats(list(self.state_space.shape), "running_stats/states")
            if self.context.normalize_actions:
                self._actions_running_stats = TfRunningStats(list(self.action_space.shape), "running_stats/actions")
            # placeholders:
            self._states_placeholder, states_input = tf_inputs([None] + list(self.state_space.shape), self.state_space.dtype, "states", cast_to_dtype=tf.float32)
            self._actions_placeholder, actions_input = tf_inputs([None] + list(self.action_space.shape), self.action_space.dtype, "actions_input", cast_to_dtype=tf.float32)
            # normalized inputs:
            states_input_normalized = self._states_running_stats.normalize(states_input, "states_input_normalize") if self.context.normalize_observations else states_input
            actions_input_normalized = self._actions_running_stats.normalize(actions_input, "actions_input_normalize") if self.context.normalize_actions else actions_input
            # actor:
            self._actor_actions = self.tf_actor(states_input_normalized, 'actor')
            # critics:
            self._critics_Qs = []
            for i in range(self.num_critics):
                critic_Q = self.tf_critic(states_input_normalized, actions_input_normalized, "critic{0}".format(i))
                self._critics_Qs.append(critic_Q)
            # actor-critic:
            actor_actions_normalized = self._actions_running_stats.normalize(self._actor_actions, "actor_actions_normalize") if self.context.normalize_actions else self._actor_actions
            self._actor_critic_Q = self.tf_critic(states_input_normalized, actor_actions_normalized, "critic0", reuse=True)
        self.params = self.get_vars()

    def check_assertions(self):
        if not hasattr(self.state_space, 'dtype'):
            self.state_space.dtype = np.float32
        if not hasattr(self.action_space, 'dtype'):
            self.action_space.dtype = np.float32
        assert hasattr(self.state_space, 'shape')
        assert hasattr(self.action_space, 'shape') and hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high')
        assert len(self.action_space.shape) == 1
        assert self.num_critics >= 1, "There should be atleast one critic, against which the actor is optimized"

    def tf_actor_activation_fn(self, x):
        y = tf.nn.tanh(x)
        y = tf_scale(y, -1, 1, self.action_space.low, self.action_space.high, "scale")
        return y

    def tf_actor(self, states, name, reuse=tf.AUTO_REUSE):
        return auto_conv_dense_net(need_conv_net(self.context.env.observation_space), states, self.context.convs, self.context.hidden_layers, self.context.activation_fn, self.action_space.shape[0], self.tf_actor_activation_fn, name, layer_norm=self.context.layer_norm, output_kernel_initializer=self.context.output_kernel_initializer, reuse=reuse)

    def tf_critic(self, states, actions, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            states_to_one_hidden = auto_conv_dense_net(need_conv_net(self.context.env.observation_space), states, self.context.convs, self.context.hidden_layers[:1], self.context.activation_fn, None, None, "one_hidden", layer_norm=self.context.layer_norm, output_kernel_initializer=None, reuse=reuse)
            one_hidden_plus_actions = tf.concat(values=[states_to_one_hidden, actions], axis=1, name="concat")
            return dense_net(one_hidden_plus_actions, self.context.hidden_layers[1:], self.context.activation_fn, 1, lambda x: x, "Q", layer_norm=self.context.layer_norm, output_kernel_initializer=self.context.output_kernel_initializer, reuse=reuse)[:, 0]

    def get_vars(self, scope=''):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{0}/{1}'.format(self.name, scope))

    def get_trainable_vars(self, scope=''):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/{1}'.format(self.name, scope))

    def get_perturbable_vars(self, scope=''):
        return list(filter(lambda var: not('LayerNorm' in var.name or 'batch_norm' in var.name or 'running_stats' in var.name), self.get_vars(scope)))

    def actions(self, states):
        return self.context.session.run(self._actor_actions, {self._states_placeholder: states})

    def Q(self, critic_id, states, actions):
        return self.context.session.run(self._critics_Qs[critic_id], {self._states_placeholder: states, self._actions_placeholder: actions})

    def setup_training(self, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            # running stats update
            if self.context.normalize_observations:
                self._update_states_running_stats = self._states_running_stats.update(self._states_placeholder[0], "update_states_running_stats")
            if self.context.normalize_actions:
                self._update_actions_running_stats = self._actions_running_stats.update(self._actions_placeholder[0], "update_actions_running_stats")
            # actor training
            self._actor_loss = -tf.reduce_mean(self._actor_critic_Q)
            actor_trainable_vars = self.get_trainable_vars('actor')
            actor_optimizer = tf.train.AdamOptimizer(self.context.actor_learning_rate, epsilon=self.context.adam_epsilon)
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
                critic_optimizer = tf.train.AdamOptimizer(self.context.learning_rate, epsilon=self.context.adam_epsilon)
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

    def train_actor(self, states):
        _, loss, Q = self.context.session.run([self._actor_train_step, self._actor_loss, self._actor_critic_Q], {self._states_placeholder: states})
        return loss, Q

    def train_critic(self, critic_id, states, actions, Q_targets):
        _, critic_loss = self.context.session.run([self._critics_train_steps[critic_id], self._critics_losses[critic_id]], feed_dict={
            self._states_placeholder: states,
            self._actions_placeholder: actions,
            self._Q_targets_placholders[critic_id]: Q_targets
        })
        return critic_loss
