from RL.common.context import Context
from RL.common.utils import RunningStats
import tensorflow as tf
from RL.common.utils import conv_net, dense_net


class RND_System:
    def __init__(self, context: Context, name):
        self.name = name
        self.context = context
        self._predicted_features_scope = "predicted_features"
        self._common_features_scope = "common_features"
        env = self.context.env  # gym.Env
        self._rewards_running_stats = RunningStats([1])
        self._states_running_stats = RunningStats(
            list(env.observation_space.shape))
        with tf.variable_scope(name):
            self._states_feed = tf.placeholder(env.observation_space.dtype, shape=[
                                               None] + list(env.observation_space.shape), name="states_feed")
            self._setup_networks(common_features_scope=self._common_features_scope,
                                 predicted_features_scope=self._predicted_features_scope)
            self._setup_reward_system()
            self._setup_training()

    def _setup_networks(self, common_features_scope="common_features", rnd_features_scope="rnd_features", predicted_features_scope="predicted_features"):
        if self.context.need_conv_net:
            self._states_representation = conv_net(
                self._states_feed, self.context.convs, self.context.activation_fn, common_features_scope)
        else:
            self._states_representation = self._states_feed
        self._rnd_features = dense_net(
            self._states_representation, self.context.rnd_layers, self.context.activation_fn, self.context.rnd_num_features, lambda x: x, rnd_features_scope)
        self._predicted_features = dense_net(
            self._states_representation, self.context.rnd_layers, self.context.activation_fn, self.context.rnd_num_features, lambda x: x, predicted_features_scope)

    def _setup_reward_system(self, scope="reward_system"):
        with tf.variable_scope(scope):
            with tf.variable_scope("squared_error"):
                error = self._predicted_features - self._rnd_features
                squared_error = tf.square(error)
            self._rewards = tf.reduce_mean(
                squared_error, axis=-1, name='rewards')

    def _get_trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="{0}/{1}".format(self.name, self._predicted_features_scope)) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="{0}/{1}".format(self.name, self._common_features_scope))

    def _setup_training(self, scope="training"):
        with tf.variable_scope(scope):
            self._optimizer = tf.train.AdamOptimizer(
                self.context.rnd_learning_rate)
            self._mse = tf.reduce_mean(self._rewards, name='mse')
            self._trainable_vars = self._get_trainable_vars()
            self._update_step = self._optimizer.minimize(
                self._mse, var_list=self._trainable_vars)

    def update_running_stats(self, states):
        self.get_rewards(states, update_stats=True)

    def get_rewards(self, states, update_stats=False):
        if update_stats:
            self._states_running_stats.update(states)
        normalized_states = self._states_running_stats.normalize(states)
        rewards = self.context.session.run(self._rewards, feed_dict={
            self._states_feed: normalized_states
        })
        if update_stats:
            self._rewards_running_stats.update(rewards)
        normalized_rewards = self._rewards_running_stats.normalize_by_std(
            rewards)
        return normalized_rewards

    def train(self, states):
        normalized_states = self._states_running_stats.normalize(states)
        return self.context.session.run([self._update_step, self._mse], feed_dict={
            self._states_feed: normalized_states
        })
