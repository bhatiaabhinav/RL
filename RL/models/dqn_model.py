from enum import Enum

import gym  # noqa: F401
import tensorflow as tf
import numpy as np

import RL
from RL.common.utils import TfRunningStats, dense_net, auto_conv_dense_net, tf_training_step, tf_safe_softmax, need_conv_net


class Brain:
    class Scopes(Enum):
        model = "model"
        states_placholder = "states"
        next_states_placeholder = "next_states"
        actions_placeholder = "actions"
        states_rms = "states_rms"
        states_embeddings = "states_embeddings"
        Q_network = "Q"
        Q_input_sensitivity = "Q_input_sensitivity"
        # transition_classifier = "transition_classifier"
        update_states_rms = "update_states_rms"
        training = "training"
        desired_Q = "desired_Q"
        Q_loss = "Q_loss"
        combined_loss = "combined_loss"
        training_step = "training_step"
        # desired_transition_classification = "desired_transition_classification"
        # tc_loss = "tc_loss"
        # tc_training_step = "tc_training_step"
        summaries = "summaries"

    def __init__(self, context: RL.Context, name, is_target, head_names):
        self.context = context
        self.name = name
        self._is_target = is_target
        self.Q_updates = 0
        self.tc_updates = 0
        self.num_heads = len(head_names)
        self.head_names = head_names
        with tf.variable_scope(self.name):
            self._setup_model()
            if not self._is_target:
                self._setup_training()

    def get_head_id(self, head_name):
        if not hasattr(self, "_head_names_to_id_map"):
            self._head_names_to_id_map = {}
            for head_id in range(self.num_heads):
                self._head_names_to_id_map[self.head_names[head_id]] = head_id
        return self._head_names_to_id_map[head_name]

    def _setup_model(self):
        with tf.variable_scope(Brain.Scopes.model.value):
            self._states_placeholder = self._tf_states_placeholder(
                Brain.Scopes.states_placholder.value)
            if self.context.normalize_observations:
                self._states_rms = TfRunningStats(list(self.context.env.observation_space.shape), Brain.Scopes.states_rms.value)
                self._states_normalized = self._states_rms.normalize(self._states_placeholder)
            else:
                self._states_normalized = self._states_placeholder
            self._states_embeddings = self._tf_states_embeddings(self._states_normalized)
            self._Qs, self._Vs = [], []
            self._Q_input_sensitivities = []
            for head_id in range(self.num_heads):
                with tf.variable_scope(self.head_names[head_id]):
                    self._Qs.append(self._tf_Q(self._states_embeddings))
                    self._Vs.append(self._tf_V(self._Qs[head_id]))
                    if not self._is_target:
                        self._Q_input_sensitivities.append(self._tf_Q_input_sensitivity(
                            self._Vs[head_id], self._states_placeholder))
            self.params = self.get_vars('')
            # if not self._is_target:
            #     self._actions_placeholder = self._tf_actions_placeholder()
            # self._next_states_placeholder = self._tf_states_placeholder(
            #     Brain.Scopes.next_states_placeholder.value)
            # self._next_states_normalized = self._states_rms.normalize(
            #     self._next_states_placeholder)
            # self._next_states_embeddings = self._tf_states_embeddings(
            #     self._next_states_normalized)
            # self._transition_classification = self._tf_transition_classification(
            #     self._states_embeddings, self._actions_placeholder, self._next_states_embeddings)

    def _setup_training(self):
        with tf.variable_scope(Brain.Scopes.training.value):
            # For updating states running stats:
            if self.context.normalize_observations:
                self._update_states_rms = self._states_rms.update(
                    self._states_placeholder[0], Brain.Scopes.update_states_rms.value)
            # Individual losses:
            self._desired_Q_placeholders, self._Q_losses, self._Q_mpes = [], [], []
            for head_id in range(self.num_heads):
                with tf.variable_scope(self.head_names[head_id]):
                    self._desired_Q_placeholders.append(
                        self._tf_desired_Q_placeholder())
                    Q_loss, mpe = self._tf_Q_loss(
                        self._Qs[head_id], self._desired_Q_placeholders[head_id])
                    self._Q_losses.append(Q_loss)
                    self._Q_mpes.append(mpe)
            # combined loss:
            self._combined_loss = 0
            with tf.variable_scope(Brain.Scopes.combined_loss.value):
                self._loss_coeffs_placeholder = tf.placeholder(
                    dtype=tf.float32, shape=[self.num_heads], name="loss_coeffs")
                for head_id in range(self.num_heads):
                    self._combined_loss += self._loss_coeffs_placeholder[head_id] * \
                        self._Q_losses[head_id]
            # optimize:
            self._optimizer = tf.train.AdamOptimizer(
                self.context.learning_rate, epsilon=self.context.adam_epsilon)
            self._trainable_vars = self.get_trainable_vars(
                Brain.Scopes.states_embeddings.value)
            for head_id in range(self.num_heads):
                self._trainable_vars += self.get_trainable_vars(
                    self.head_names[head_id] + "/" + Brain.Scopes.Q_network.value)
            self._training_step = tf_training_step(self._combined_loss, self._trainable_vars, self._optimizer,
                                                   self.context.l2_reg, self.context.clip_gradients, Brain.Scopes.training_step.value)
            # self._tc_optimizer = tf.train.AdamOptimizer(self.context.learning_rate)
            # self._desired_transition_classification = self._tf_desired_transition_classification_placeholder()
            # self._transition_classification_loss = self._tf_transition_classification_loss(
            #     self._transition_classification, self._desired_transition_classification)
            # self._tc_trainable_vars = self.get_trainable_vars(
            #     Brain.Scopes.states_embeddings.value) + self.get_trainable_vars(Brain.Scopes.transition_classifier.value)
            # self._tc_training_step = tf_training_step(self._transition_classification_loss, self._tc_trainable_vars,
            #                                           self._tc_optimizer, self.context.l2_reg, self.context.clip_gradients, Brain.Scopes.tc_training_step.value)

    def _tf_states_placeholder(self, name):
        with tf.variable_scope(name):
            placeholder = tf.placeholder(dtype=self.context.env.observation_space.dtype, shape=[
                                         None] + list(self.context.env.observation_space.shape))
            states = tf.cast(placeholder, tf.float32)
            # states = tf_scale(states, self.context.env.observation_space.low,
            #                   self.context.env.observation_space.high, -1, 1, "scale_minus1_to_1")
            return states

    def _tf_actions_placeholder(self):
        with tf.variable_scope(Brain.Scopes.actions_placeholder.value):
            placeholder = tf.placeholder(dtype=tf.float32, shape=[
                                         None] + [self.context.env.action_space.n])
            return placeholder

    def _tf_states_embeddings(self, inputs, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(Brain.Scopes.states_embeddings.value, reuse=reuse):
            return auto_conv_dense_net(need_conv_net(self.context.env.observation_space), inputs, self.context.convs, self.context.states_embedding_hidden_layers, self.context.activation_fn, None, None, "conv_dense_net", reuse=reuse)

    def _tf_Q(self, inputs, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(Brain.Scopes.Q_network.value, reuse=reuse):
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

    def _tf_V(self, Q):
        with tf.variable_scope('V'):
            return tf.reduce_max(Q, axis=-1)

    def _tf_Q_input_sensitivity(self, V, inputs):
        with tf.variable_scope(Brain.Scopes.Q_input_sensitivity.value):
            grads = tf.gradients(V, [inputs])[0]
            abs_grads = tf.abs(grads)
            dims_to_reduce = list(range(len(grads.get_shape().as_list())))[1:]
            scaled_abs_grads = abs_grads / \
                tf.reduce_max(abs_grads, axis=dims_to_reduce)
            return scaled_abs_grads

    def _tf_transition_classification(self, states, actions, next_states):
        """returns the probability of next_state coming after state-action pair. Expects flattened inputs"""
        with tf.variable_scope(Brain.Scopes.transition_classifier.value):
            concated = tf.concat(
                values=[states, actions, next_states], axis=-1, name="concat")
            hidden_layers = [self.context.states_embedding_hidden_layers[-1]]
            classification_logits = dense_net(
                concated, hidden_layers, self.context.activation_fn, 2, lambda x: x, 'dense_net')
            classification_logits = tf_safe_softmax(
                classification_logits, 'softmax')
            return classification_logits

    def _tf_desired_Q_placeholder(self):
        return tf.placeholder(dtype='float32', shape=[None, self.context.env.action_space.n], name=Brain.Scopes.desired_Q.value)

    def _tf_Q_loss(self, Q, desired_Q):
        with tf.variable_scope(Brain.Scopes.Q_loss.value):
            error = Q - desired_Q
            percentage_error = 100 * tf.abs(error) / (tf.abs(desired_Q) + 1e-3)
            # squared_error = tf.square(error)
            # mse = tf.reduce_mean(squared_error, name="mse")
            mpe = tf.reduce_mean(percentage_error, name="mpe")
            huber_loss = tf.losses.huber_loss(desired_Q, Q, scope="huber_loss")
            return huber_loss, mpe

    def _tf_desired_transition_classification_placeholder(self):
        return tf.placeholder(dtype='float32', shape=[None, 2], name=Brain.Scopes.desired_transition_classification.value)

    def _tf_transition_classification_loss(self, tc, desired_tc):
        with tf.variable_scope(Brain.Scopes.tc_loss.value):
            # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tc, labels=desired_tc))
            return -tf.reduce_mean((desired_tc[:, 0] * tf.log(tc[:, 0]) + desired_tc[:, 1] * tf.log(tc[:, 1])))

    def _get_fetches(self, fetch_list, head_names):
        if head_names is None:
            return fetch_list
        else:
            fetches = []
            for head_name in head_names:
                fetches.append(fetch_list[self.get_head_id(head_name)])
            return fetches

    def get_Q(self, states, head_names=None):
        return self.context.session.run(self._get_fetches(self._Qs, head_names), feed_dict={
            self._states_placeholder: states
        })

    def get_maxQ(self, states, head_names=None):
        return self.context.session.run(self._get_fetches(self._Vs, head_names), feed_dict={
            self._states_placeholder: states
        })

    def get_argmax_Q(self, states, head_names=None):
        Qs = self.get_Q(states, head_names=head_names)
        argmax_Qs = []
        for Q in Qs:
            argmax_Qs.append(np.argmax(Q, axis=-1))
        return argmax_Qs

    def get_Q_input_sensitivity(self, states, head_names=None):
        sensitivities = self.context.session.run(self._get_fetches(self._Q_input_sensitivities, head_names), feed_dict={
            self._states_placeholder: states
        })
        return sensitivities

    def get_vars(self, scope):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{0}/{1}/{2}'.format(self.name, Brain.Scopes.model.value, scope))

    def get_trainable_vars(self, scope):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/{1}/{2}'.format(self.name, Brain.Scopes.model.value, scope))

    def get_perturbable_vars(self, scope):
        return [v for v in self.get_vars(scope) if not('LayerNorm' in v.name or 'batch_norm' in v.name or 'rms' in v.name)]

    def update_running_stats(self, states):
        for i in range(len(states)):
            state = states[i]
            self.context.session.run(self._update_states_rms, feed_dict={
                self._states_placeholder: [state]
            })

    def train(self, mb_states, *list_mb_desiredQ_per_head, loss_coeffs_per_head=None):
        self.Q_updates += 1
        if loss_coeffs_per_head is None:
            loss_coeffs_per_head = [1.0] * self.num_heads
        fetches = [self._training_step, self._combined_loss]
        fetches += self._get_fetches(self._Q_losses, None)
        fetches += self._get_fetches(self._Q_mpes, None)
        fetches += self._get_fetches(self._Vs, None)
        feed_dict = {
            self._states_placeholder: mb_states,
            self._loss_coeffs_placeholder: loss_coeffs_per_head
        }
        for head_id in range(self.num_heads):
            feed_dict[self._desired_Q_placeholders[head_id]
                      ] = list_mb_desiredQ_per_head[head_id]
        results = self.context.session.run(fetches, feed_dict=feed_dict)
        combined_loss = results[1]
        Q_losses = results[2:2 + self.num_heads]
        Q_mpes = results[2 + self.num_heads:2 + 2 * self.num_heads]
        Vs = results[2 + 2 * self.num_heads:]
        for head_name, Q_loss, Q_mpe, V in zip(self.head_names, Q_losses, Q_mpes, Vs):
            if self.Q_updates % 10 == 0:
                mb_av_V_summary_name = "{0}/{1}/mb_av_V".format(self.name, head_name)
                Q_loss_summary_name = "{0}/{1}/Q_loss".format(self.name, head_name)
                Q_mpe_summary_name = "{0}/{1}/Q_mpe".format(self.name, head_name)
                RL.stats.record_append(mb_av_V_summary_name, np.mean(V))
                RL.stats.record_append(Q_loss_summary_name, Q_loss)
                RL.stats.record_append(Q_mpe_summary_name, Q_mpe)
                RL.stats.record_append("Q Updates", self.Q_updates)
        return combined_loss, Q_losses, Q_mpes

    def train_transitions(self, mb_states, mb_actions, mb_next_states, mb_desired_tc):
        self.tc_updates += 1
        _, loss, tc = self.context.session.run([self._tc_training_step, self._transition_classification_loss, self._transition_classification], feed_dict={
            self._states_placeholder: mb_states,
            self._actions_placeholder: mb_actions,
            self._next_states_placeholder: mb_next_states,
            self._desired_transition_classification: mb_desired_tc
        })
        if self.tc_updates == 1:
            self.context.summaries.setup_scalar_summaries(
                ["mb_av_tc", "tc_loss", "tc_accuracy"])
        if self.tc_updates % 10 == 0:
            tc_accuracy = np.mean(np.equal(
                np.argmax(mb_desired_tc, axis=-1), np.argmax(tc, axis=-1)).astype(np.int))
            self.context.log_summary(
                {"mb_av_tc": np.mean(np.argmax(tc, -1)), "tc_loss": loss, "tc_accuracy": tc_accuracy}, self.tc_updates)
