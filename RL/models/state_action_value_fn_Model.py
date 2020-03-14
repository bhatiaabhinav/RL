import RL
import tensorflow as tf
from RL.common.utils import tf_inputs, tf_training_step, need_conv_net, TfRunningStats, conv_net, dense_net


class StateActionValueFnModel:
    def __init__(self, context: RL.Context, name, reuse=tf.AUTO_REUSE):
        self.context = context
        self.session = self.context.session
        self.name = name
        self.state_space = self.context.env.observation_space
        self.action_space = self.context.env.action_space
        with tf.variable_scope(name, reuse=reuse):
            # running stats:
            if self.context.normalize_observations:
                self._states_running_stats = TfRunningStats(list(self.state_space.shape), "running_stats/states", reuse=reuse)
            if self.context.normalize_actions:
                self._actions_running_stats = TfRunningStats(list(self.action_space.shape), "running_stats/actions", reuse=reuse)
            # placeholders:
            self._states_placeholder, states_input = tf_inputs([None] + list(self.state_space.shape), self.state_space.dtype, "states", cast_to_dtype=tf.float32)
            self._actions_placeholder, actions_input = tf_inputs([None] + list(self.action_space.shape), self.action_space.dtype, "actions", cast_to_dtype=tf.float32)
            # normalized inputs:
            states_input_normalized = self._states_running_stats.normalize(states_input, "states_input_normalize") if self.context.normalize_observations else states_input
            actions_input_normalized = self._actions_running_stats.normalize(actions_input, "actions_input_normalize") if self.context.normalize_actions else actions_input
            self.action_value_tensor = self.tf_action_value_fn(states_input_normalized, actions_input_normalized, 'state_action_val', reuse=reuse)
        self.params = self.get_vars('')

    def tf_action_value_fn(self, states, actions, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            if need_conv_net(self.state_space):
                states = conv_net(states, self.context.convs, self.context.activation_fn, 'conv', reuse=reuse)
            states_actions = tf.concat(values=[states, actions], axis=-1)
            return dense_net(states_actions, self.context.hidden_layers, self.context.activation_fn, 1, lambda x: x, "dense", layer_norm=self.context.layer_norm, output_kernel_initializer=self.context.output_kernel_initializer, reuse=reuse)[:, 0]

    def setup_training(self, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            self._V_targets_placholder, _ = tf_inputs([None], tf.float32, "Target_V")
            self._loss = tf.losses.mean_squared_error(self._V_targets_placholder, self._V)
            self._optimizer = tf.train.AdamOptimizer(self.context.learning_rate)
            V_trainable_vars = self.get_trainable_vars(self._V_vars_scope)
            assert len(V_trainable_vars) > 0, "No vars to train!"
            self._train_step = tf_training_step(self._loss, V_trainable_vars, self._optimizer, self.context.l2_reg, self.context.clip_gradients, "train_step")
            return self._train_step

    def setup_training(self, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            # running stats update
            if self.context.normalize_observations:
                self._update_states_running_stats = self._states_running_stats.update(self._states_placeholder[0], "update_states_running_stats")
            if self.context.normalize_actions:
                self._update_actions_running_stats = self._actions_running_stats.update(self._actions_placeholder[0], "update_actions_running_stats")
            # critics training
            if self.num_critics:
                critics_trainable_vars = self.get_trainable_vars(*('critic{0}'.format(i) for i in range(self.num_critics)))
                critics_optimizer = tf.train.AdamOptimizer(self.context.learning_rate, epsilon=self.context.adam_epsilon)
                assert len(critics_trainable_vars) > 0, "No vars to train in critics"
                self._critics_train_step = tf_training_step(self._critics_loss, critics_trainable_vars, critics_optimizer, self.context.l2_reg, self.context.clip_gradients, "critics_train_step")
            # valuefns training
            if self.num_valuefns:
                valuefns_trainable_vars = self.get_trainable_vars(*('valuefn{0}'.format(i) for i in range(self.num_valuefns)))
                valuefns_optimizer = tf.train.AdamOptimizer(self.context.learning_rate, epsilon=self.context.adam_epsilon)
                assert len(valuefns_trainable_vars) > 0, "No vars to train in valuefns"
                self._valuefns_train_step = tf_training_step(self._valuefns_loss, valuefns_trainable_vars, valuefns_optimizer, self.context.l2_reg, self.context.clip_gradients, "valuefns_train_step")

    def setup_saving_loading_ops(self, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            params = self.get_vars('')
            self._load_placeholders = []
            self._load_ops = []
            for p in params:
                p_placeholder = tf.placeholder(shape=p.shape.as_list(), dtype=tf.float32)
                self._load_placeholders.append(p_placeholder)
                self._load_ops.append(p.assign(p_placeholder))

    def get_vars(self, scope):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{0}/{1}'.format(self.name, scope))

    def get_trainable_vars(self, scope):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/{1}'.format(self.name, scope))

    def get_V(self, states):
        return self.session.run(self._V, feed_dict={self._states_placeholder: states})

    def train_V(self, states, V_targets):
        _, loss = self.session.run([self._train_step, self._loss], feed_dict={
            self._states_placeholder: states,
            self._V_targets_placholder: V_targets
        })
        return loss
