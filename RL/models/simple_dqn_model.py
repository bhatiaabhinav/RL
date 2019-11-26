import gym  # noqa: F401
import tensorflow as tf
import numpy as np
import sys

import RL
from RL.common.utils import TfRunningStats, dense_net, auto_conv_dense_net, tf_training_step, need_conv_net, tf_scale, tf_safe_softmax


class Brain:
    def __init__(self, context: RL.Context, name, is_target):
        self.context = context
        self.name = name
        self._is_target = is_target
        self.Q_updates = 0
        with tf.variable_scope(self.name):
            self._setup_model()
            if not self._is_target:
                self._setup_training()

    def _setup_model(self):
        with tf.variable_scope("model"):
            self._states_placeholder = self._tf_states_placeholder("states")
            if self.context.normalize_observations:
                self._states_rms = TfRunningStats(list(self.context.env.observation_space.shape), "states_rms")
                self._states_normalized = self._states_rms.normalize(self._states_placeholder)
            else:
                self._states_normalized = self._states_placeholder
            self._states_embeddings = self._tf_states_embeddings(self._states_normalized, "states_embeddings")
            self._Q = self._tf_Q(self._states_embeddings, "Q")
            # self._Q = self._tf_modular2_Q(self._states_embeddings, "Q")
            self._V = self._tf_V(self._Q, "V")
            if not self._is_target:
                self._Q_input_sensitivity = self._tf_Q_input_sensitivity(self._V, self._states_placeholder, "Q_input_sensitivity")
            self.params = self.get_vars('')

    def _setup_training(self):
        with tf.variable_scope("training"):
            # For updating states running stats:
            if self.context.normalize_observations:
                self._update_states_rms = self._states_rms.update(self._states_placeholder[0], "update_states_rms")
            # losses:
            self._desired_Q_placeholder = self._tf_desired_Q_placeholder("desired_Q")
            self._Q_loss, self._Q_mpe = self._tf_Q_loss(self._Q, self._desired_Q_placeholder, "Q_loss")
            self._attention_entropy = tf.reduce_mean(tf.reduce_sum(- self._attention * tf.log(self._attention), -1))
            # optimize:
            self._optimizer = tf.train.AdamOptimizer(self.context.learning_rate, epsilon=self.context.adam_epsilon)
            self._trainable_vars = self.get_trainable_vars("states_embeddings")
            self._trainable_vars += self.get_trainable_vars("Q")
            self._training_step = tf_training_step(self._Q_loss - 0.01 * self._attention_entropy, self._trainable_vars, self._optimizer,
                                                   self.context.l2_reg, self.context.clip_gradients, "training_step")

    def _tf_states_placeholder(self, name):
        with tf.variable_scope(name):
            placeholder = tf.placeholder(dtype=self.context.env.observation_space.dtype, shape=[
                                         None] + list(self.context.env.observation_space.shape))
            states = tf.cast(placeholder, tf.float32)
            states = tf_scale(states, self.context.env.observation_space.low,
                              self.context.env.observation_space.high, -1, 1, "scale_minus1_to_1")
            return states

    def _tf_actions_placeholder(self):
        with tf.variable_scope("actions"):
            return tf.placeholder(dtype=tf.float32, shape=[None] + [self.context.env.action_space.n])

    def _tf_states_embeddings(self, inputs, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            o = self.context.env.observation_space  # type: gym.spaces.Box
            if need_conv_net(o) and o.shape[0] < o.shape[1]:
                inputs = tf.transpose(inputs, [0, 2, 3, 1])
            return auto_conv_dense_net(need_conv_net(o), inputs, self.context.convs, self.context.states_embedding_hidden_layers, self.context.activation_fn, None, None, "conv_dense_net", reuse=reuse)

    def _tf_Q(self, inputs, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            if not self.context.dueling_dqn:
                Q = dense_net(inputs, self.context.hidden_layers, self.context.activation_fn,
                              self.context.env.action_space.n, lambda x: x, 'dense_net', layer_norm=self.context.layer_norm, output_kernel_initializer=self.context.output_kernel_initializer, reuse=reuse)
            else:
                A_dueling = dense_net(inputs, self.context.hidden_layers, self.context.activation_fn,
                                      self.context.env.action_space.n, lambda x: x, 'A_dueling', layer_norm=self.context.layer_norm, output_kernel_initializer=self.context.output_kernel_initializer, reuse=reuse)
                V_dueling = dense_net(inputs, self.context.hidden_layers,
                                      self.context.activation_fn, 1, lambda x: x, 'V_dueling', layer_norm=self.context.layer_norm, output_kernel_initializer=self.context.output_kernel_initializer, reuse=reuse)
                Q = V_dueling + A_dueling - \
                    tf.reduce_mean(A_dueling, axis=1, keepdims=True)
            return Q

    def _tf_modular_Q(self, inputs, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            n_modules = 32
            attention_logits = dense_net(inputs, [64], self.context.activation_fn, n_modules, lambda x: x, 'attention', layer_norm=self.context.layer_norm, output_kernel_initializer=self.context.output_kernel_initializer, reuse=reuse)
            attention = tf_safe_softmax(attention_logits, "softmax")  # [batch, m]
            attention = tf.expand_dims(attention, 1)  # [batch, 1, m]
            Qs = dense_net(inputs, self.context.hidden_layers, self.context.activation_fn, self.context.env.action_space.n * n_modules, lambda x: x, 'dense_net', layer_norm=self.context.layer_norm, output_kernel_initializer=self.context.output_kernel_initializer, reuse=reuse)  # [batch, m * n]
            Qs = tf.reshape(Qs, shape=[-1, n_modules, self.context.env.action_space.n])  # [batch, m, n]
            Q = tf.matmul(attention, Qs)
            return Q[:, 0, :]

    def _tf_modular2_Q(self, inputs, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            n_modules = 16
            attention_logits = dense_net(inputs, [n_modules * 4], self.context.activation_fn, n_modules, lambda x: x, 'attention', layer_norm=self.context.layer_norm, output_kernel_initializer=self.context.output_kernel_initializer, reuse=reuse)
            attention = tf_safe_softmax(attention_logits, "softmax")  # [batch, m]
            self._attention = attention
            # attention = tf.expand_dims(attention, 1)  # [batch, 1, m]
            Q = 0
            for i in range(n_modules):
                Q += attention[:, i:i + 1] * dense_net(inputs, list(np.asarray(self.context.hidden_layers) // n_modules), self.context.activation_fn, self.context.env.action_space.n, lambda x: x, 'dense_net_' + str(i), layer_norm=self.context.layer_norm, output_kernel_initializer=self.context.output_kernel_initializer, reuse=reuse)  # [batch, n]
            return Q

    def _tf_V(self, Q, name="V"):
        with tf.variable_scope(name):
            return tf.reduce_max(Q, axis=-1)

    def _tf_Q_input_sensitivity(self, V, inputs, name):
        with tf.variable_scope(name):
            grads = tf.gradients(V, [inputs])[0]
            abs_grads = tf.abs(grads)
            dims_to_reduce = list(range(len(grads.get_shape().as_list())))[1:]
            scaled_abs_grads = abs_grads / tf.reduce_max(abs_grads, axis=dims_to_reduce)
            return scaled_abs_grads

    def _tf_desired_Q_placeholder(self, name):
        return tf.placeholder(dtype='float32', shape=[None, self.context.env.action_space.n], name=name)

    def _tf_Q_loss(self, Q, desired_Q, name):
        with tf.variable_scope(name):
            error = Q - desired_Q
            percentage_error = 100 * tf.abs(error) / (tf.abs(desired_Q) + 1e-3)
            mpe = tf.reduce_mean(percentage_error, name="mpe")
            huber_loss = tf.losses.huber_loss(desired_Q, Q, scope="huber_loss")
            return huber_loss, mpe

    def Q(self, states):
        return self.context.session.run(self._Q, feed_dict={self._states_placeholder: states})

    def V(self, states):
        return self.context.session.run(self._V, feed_dict={self._states_placeholder: states})

    def argmax_Q(self, states):
        Q = self.Q(states)
        return np.argmax(Q, axis=-1)

    def Q_input_sensitivity(self, states):
        return self.context.session.run(self._Q_input_sensitivity, feed_dict={self._states_placeholder: states})

    def get_vars(self, scope):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{0}/{1}/{2}'.format(self.name, "model", scope))

    def get_trainable_vars(self, scope):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/{1}/{2}'.format(self.name, "model", scope))

    def get_perturbable_vars(self, scope):
        return [v for v in self.get_vars(scope) if not('LayerNorm' in v.name or 'batch_norm' in v.name or 'rms' in v.name)]

    def update_running_stats(self, states):
        for i in range(len(states)):
            state = states[i]
            self.context.session.run(self._update_states_rms, feed_dict={
                self._states_placeholder: [state]
            })

    def train(self, mb_states, mb_desiredQ):
        self.Q_updates += 1
        _, Q_loss, Q_mpe, V, att, att_ent = self.context.session.run([self._training_step, self._Q_loss, self._Q_mpe, self._V, self._attention, self._attention_entropy], feed_dict={
            self._states_placeholder: mb_states,
            self._desired_Q_placeholder: mb_desiredQ
        })
        if self.Q_updates % 100 == 0:
            RL.stats.record_append("{0}/mb_av_V".format(self.name), np.mean(V))
            RL.stats.record_append("{0}/Q_loss".format(self.name), Q_loss)
            RL.stats.record_append("{0}/Q_mpe".format(self.name), Q_mpe)
            RL.stats.record_append("Q Updates", self.Q_updates)
            RL.stats.record("att_hist", np.argmax(att, -1))
            RL.stats.record_append("att_ent", att_ent)
        return Q_loss, Q_mpe
