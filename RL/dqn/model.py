from enum import Enum

import gym  # noqa: F401
import tensorflow as tf
import numpy as np
import os
import joblib

from RL.common.context import Context
from RL.common.utils import TfRunningStats, dense_net, auto_conv_dense_net, tf_training_step, tf_scale
from RL.common import logger


class Brain:
    class Scopes(Enum):
        model = "model"
        states_placholder = "states"
        next_states_placeholder = "next_states"
        states_rms = "states_rms"
        states_embeddings = "states_embeddings"
        Q_network = "Q"
        Q_input_sensitivity = "Q_input_sensitivity"
        transition_classifier = "transition_classifier"
        update_states_rms = "update_states_rms"
        training = "training"
        desired_Q = "desired_Q"
        Q_loss = "Q_loss"
        Q_training_step = "Q_training_step"
        desired_transition_classification = "desired_transition_classification"
        tc_loss = "tc_loss"
        tc_training_step = "tc_training_step"
        summaries = "summaries"

    def __init__(self, context: Context, name, is_target):
        self.context = context
        self.name = name
        self._is_target = is_target
        self.Q_updates = 0
        self.tc_updates = 0
        with tf.variable_scope(self.name):
            self._setup_model()
            if not self._is_target:
                self._setup_training()
                self._setup_saving_loading_ops()

    def _setup_model(self):
        with tf.variable_scope(Brain.Scopes.model.value):
            self._states_placeholder = self._tf_states_placeholder(
                Brain.Scopes.states_placholder.value)
            self._states_rms = TfRunningStats(
                list(self.context.env.observation_space.shape), Brain.Scopes.states_rms.value)
            self._states_normalized = self._states_rms.normalize(
                self._states_placeholder)
            self._states_embeddings = self._tf_states_embeddings(
                self._states_normalized)
            self._Q = self._tf_Q(self._states_embeddings)
            self._V = self._tf_V(self._Q)
            self._greedy_action = self._tf_greedy_action(self._Q)
            self._Q_input_sensitivity = self._tf_Q_input_sensitivity(self._V, self._states_placeholder)
            if not self._is_target:
                self._next_states_placeholder = self._tf_states_placeholder(
                    Brain.Scopes.next_states_placeholder.value)
                self._next_states_normalized = self._states_rms.normalize(
                    self._next_states_placeholder)
                self._next_states_embeddings = self._tf_states_embeddings(
                    self._next_states_normalized)
                self._transition_classification = self._tf_transition_classification(
                    self._states_embeddings, self._next_states_embeddings)

    def _setup_training(self):
        with tf.variable_scope(Brain.Scopes.training.value):
            self._update_states_rms = self._states_rms.update(
                self._states_placeholder[0], Brain.Scopes.update_states_rms.value)
            self._Q_optimizer = tf.train.AdamOptimizer(
                self.context.learning_rate)
            self._desired_Q_placeholder = self._tf_desired_Q_placeholder()
            self._Q_trainable_vars = self.get_trainable_vars(
                Brain.Scopes.states_embeddings.value) + self.get_trainable_vars(Brain.Scopes.Q_network.value)
            self._Q_loss = self._tf_Q_loss(
                self._Q, self._desired_Q_placeholder)
            self._Q_training_step = tf_training_step(
                self._Q_loss, self._Q_trainable_vars, self._Q_optimizer, self.context.l2_reg, self.context.clip_gradients, Brain.Scopes.Q_training_step.value)
            self._tc_optimizer = tf.train.AdamOptimizer(self.context.learning_rate)
            self._desired_transition_classification = self._tf_desired_transition_classification_placeholder()
            self._transition_classification_loss = self._tf_transition_classification_loss(
                self._transition_classification, self._desired_transition_classification)
            self._tc_trainable_vars = self.get_trainable_vars(
                Brain.Scopes.states_embeddings.value) + self.get_trainable_vars(Brain.Scopes.transition_classifier.value)
            self._tc_training_step = tf_training_step(self._transition_classification_loss, self._tc_trainable_vars,
                                                      self._tc_optimizer, self.context.l2_reg, self.context.clip_gradients, Brain.Scopes.tc_training_step.value)

    def _setup_saving_loading_ops(self):
        with tf.variable_scope('saving_loading_ops'):
            params = self.get_vars('')
            self._load_placeholders = []
            self._load_ops = []
            for p in params:
                p_placeholder = tf.placeholder(
                    shape=p.shape.as_list(), dtype=tf.float32)
                self._load_placeholders.append(p_placeholder)
                self._load_ops.append(p.assign(p_placeholder))

    def _tf_states_placeholder(self, name):
        with tf.variable_scope(name):
            placeholder = tf.placeholder(dtype=self.context.env.observation_space.dtype, shape=[None] + list(self.context.env.observation_space.shape))
            states = tf.cast(placeholder, tf.float32)
            states = tf_scale(states, self.context.env.observation_space.low, self.context.env.observation_space.high, -1, 1, "scale_minus1_to_1")
            return states

    def _tf_states_embeddings(self, inputs):
        with tf.variable_scope(Brain.Scopes.states_embeddings.value, reuse=tf.AUTO_REUSE):
            return auto_conv_dense_net(self.context.need_conv_net, inputs, self.context.convs, self.context.states_embedding_hidden_layers, self.context.activation_fn, None, None, "conv_dense_net")

    def _tf_Q(self, inputs):
        with tf.variable_scope(Brain.Scopes.Q_network.value, reuse=tf.AUTO_REUSE):
            if not self.context.dueling_dqn:
                Q = dense_net(inputs, self.context.Q_hidden_layers, self.context.activation_fn,
                              self.context.env.action_space.n, lambda x: x, 'dense_net')
            else:
                A_dueling = dense_net(inputs, self.context.Q_hidden_layers, self.context.activation_fn,
                                      self.context.env.action_space.n, lambda x: x, 'A_dueling')
                V_dueling = dense_net(inputs, self.context.Q_hidden_layers, self.context.activation_fn,
                                      self.context.env.action_space.n, lambda x: x, 'V_dueling')
                Q = V_dueling + A_dueling - \
                    tf.reduce_mean(A_dueling, axis=1, keepdims=True)
            return Q

    def _tf_greedy_action(self, Q):
        with tf.variable_scope('greedy_action'):
            return tf.argmax(Q, axis=1)

    def _tf_V(self, Q):
        with tf.variable_scope('V'):
            return tf.reduce_max(Q, axis=1)

    def _tf_Q_input_sensitivity(self, V, inputs):
        with tf.variable_scope(Brain.Scopes.Q_input_sensitivity.value):
            grads = tf.gradients(V, [inputs])[0]
            abs_grads = tf.abs(grads)
            dims_to_reduce = list(range(len(grads.get_shape().as_list())))[1:]
            scaled_abs_grads = abs_grads / tf.reduce_max(abs_grads, axis=dims_to_reduce)
            return scaled_abs_grads

    def _tf_transition_classification(self, states, next_states):
        """returns the probability of next_state coming after state. Expects flattened inputs"""
        with tf.variable_scope(Brain.Scopes.transition_classifier.value):
            concated = tf.concat(values=[states, next_states], axis=-1, name="concat")
            hidden_layers = [self.context.states_embedding_hidden_layers[-1]]
            classification_logits = dense_net(
                concated, hidden_layers, self.context.activation_fn, 2, lambda x: x, 'dense_net')
            return classification_logits

    def _tf_desired_Q_placeholder(self):
        return tf.placeholder(dtype='float32', shape=[None, self.context.env.action_space.n], name=Brain.Scopes.desired_Q.value)

    def _tf_Q_loss(self, Q, desired_Q):
        with tf.variable_scope(Brain.Scopes.Q_loss.value):
            error = Q - desired_Q
            squared_error = tf.square(error)
            mse = tf.reduce_mean(squared_error)
            return mse

    def _tf_desired_transition_classification_placeholder(self):
        return tf.placeholder(dtype='float32', shape=[None, 2], name=Brain.Scopes.desired_transition_classification.value)

    def _tf_transition_classification_loss(self, tc, desired_tc):
        with tf.variable_scope(Brain.Scopes.tc_loss.value):
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tc, labels=desired_tc))

    def get_Q(self, states, update_stats=False):
        return self.context.session.run(self._Q, feed_dict={
            self._states_placeholder: states
        })

    def get_V(self, states):
        return self.context.session.run(self._V, feed_dict={
            self._states_placeholder: states
        })

    def get_action(self, states):
        return self.context.session.run(self._greedy_action, feed_dict={
            self._states_placeholder: states
        })

    def get_Q_input_sensitivity(self, states):
        sensitivities = self.context.session.run(self._Q_input_sensitivity, feed_dict={
            self._states_placeholder: states
        })
        return sensitivities

    def get_vars(self, scope):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{0}/{1}/{2}'.format(self.name, Brain.Scopes.model.value, scope))

    def get_trainable_vars(self, scope):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/{1}/{2}'.format(self.name, Brain.Scopes.model.value, scope))

    def get_perturbable_vars(self, scope):
        return [v for v in self.get_vars(scope) if not('LayerNorm' in v.name or 'batch_norm' in v.name or 'rms' in v.name)]

    def _tf_scope_copy_to(self, other_brain, scope, name, soft_copy=False):
        with tf.variable_scope(name):
            other_brain = other_brain  # type: Brain
            my_vars = self.get_vars(scope)
            other_brain_vars = other_brain.get_vars(scope)
            assert len(my_vars) == len(
                other_brain_vars), "Something is wrong! Both brains should have same number of vars in scope {0}".format(scope)
            assert len(
                my_vars) > 0, "There are no variables in scope {0}!".format(scope)
            copy_ops = []
            for my_var, other_var in zip(my_vars, other_brain_vars):
                copy_op = tf.assign(other_var, my_var * self.context.tau + (
                    1 - self.context.tau) * other_var) if soft_copy else tf.assign(other_var, my_var)
                copy_ops.append(copy_op)
            return copy_ops

    def tf_copy_to(self, other_brain, soft_copy=False, name='copy_brain_ops'):
        with tf.variable_scope(name):
            Q_copy = self._tf_scope_copy_to(
                other_brain, Brain.Scopes.Q_network.value, "Q_params_copy", soft_copy=soft_copy)
            embeddings_copy = self._tf_scope_copy_to(
                other_brain, Brain.Scopes.states_embeddings.value, "states_embs_params_copy", soft_copy=False)
            rms_copy = self._tf_scope_copy_to(
                other_brain, Brain.Scopes.states_rms.value, "states_rms_vars_copy", soft_copy=False)
            all_ops = Q_copy + embeddings_copy + rms_copy
            return all_ops

    def update_running_stats(self, states):
        for i in range(len(states)):
            state = states[i]
            self.context.session.run(self._update_states_rms, feed_dict={
                self._states_placeholder: [state]
            })

    def train(self, mb_states, mb_desiredQ):
        self.Q_updates += 1
        _, loss, V = self.context.session.run([self._Q_training_step, self._Q_loss, self._V], feed_dict={
            self._states_placeholder: mb_states,
            self._desired_Q_placeholder: mb_desiredQ
        })
        if self.Q_updates == 1:
            self.context.summaries.setup_scalar_summaries(
                ["mb_av_V", "Q_loss"])
        if self.Q_updates % 10 == 0:
            self.context.log_summary(
                {"mb_av_V": np.mean(V), "Q_loss": loss}, self.Q_updates)
        return _, loss

    def train_transitions(self, mb_states, mb_next_states, mb_desired_tc):
        self.tc_updates += 1
        _, loss, tc = self.context.session.run([self._tc_training_step, self._transition_classification_loss, self._transition_classification], feed_dict={
            self._states_placeholder: mb_states,
            self._next_states_placeholder: mb_next_states,
            self._desired_transition_classification: mb_desired_tc
        })
        if self.tc_updates == 1:
            self.context.summaries.setup_scalar_summaries(
                ["mb_av_tc", "tc_loss"])
        if self.tc_updates % 10 == 0:
            self.context.log_summary(
                {"mb_av_tc": np.mean(np.argmax(tc, -1)), "tc_loss": loss}, self.tc_updates)

    def save(self, save_path=None, suffix=''):
        if save_path is None:
            save_path = os.path.join(logger.get_dir(), "saved_models", "model{0}".format(suffix))
        params = self.context.session.run(self.get_vars(''))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(params, save_path)

    def load(self, load_path):
        params = joblib.load(load_path)
        feed_dict = {}
        for p, p_placeholder in zip(params, self._load_placeholders):
            feed_dict[p_placeholder] = p
        self.context.session.run(self._load_ops, feed_dict=feed_dict)
