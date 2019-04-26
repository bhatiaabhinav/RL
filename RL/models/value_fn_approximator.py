import RL
import tensorflow as tf
from RL.common.utils import tf_inputs, auto_conv_dense_net, tf_training_step


class ValueFunctionApproximator:
    def __init__(self, context: RL.Context, name, input_shape, input_dtype, need_conv=False, reuse=tf.AUTO_REUSE):
        self.context = context
        self.session = self.context.session
        self.name = name
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.need_conv = need_conv
        self._V_vars_scope = "V"
        with tf.variable_scope(name, reuse=reuse):
            self._states_placeholder, inputs = tf_inputs([None] + self.input_shape, self.input_dtype, "inputs", cast_to_dtype=tf.float32)
            self._V = auto_conv_dense_net(self.need_conv, inputs, self.context.convs, self.context.hidden_layers, self.context.activation_fn, 1, lambda x: x, self._V_vars_scope, output_kernel_initializer=tf.initializers.random_uniform(-1e-3, 1e-3), reuse=reuse)[0]
        self.params = self.get_vars('')

    def setup_training(self, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            self._V_targets_placholder, _ = tf_inputs([None], tf.float32, "Target_V")
            self._loss = tf.losses.mean_squared_error(self._V_targets_placholder, self._V)
            self._optimizer = tf.train.AdamOptimizer(self.context.learning_rate)
            V_trainable_vars = self.get_trainable_vars(self._V_vars_scope)
            assert len(V_trainable_vars) > 0, "No vars to train!"
            self._train_step = tf_training_step(self._loss, V_trainable_vars, self._optimizer, self.context.l2_reg, self.context.clip_gradients, "train_step")
            return self._train_step

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
