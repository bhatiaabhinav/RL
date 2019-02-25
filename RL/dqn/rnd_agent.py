from collections import deque
from enum import Enum
from typing import List  # noqa: F401

import gym  # noqa: F401
import numpy as np
import tensorflow as tf

from RL.common import logger
from RL.common.atari_wrappers import wrap_atari
from RL.common.context import (
    Agent, Context, PygletLoop, RLRunner, SeedingAgent, SimpleRenderingAgent)
from RL.common.utils import ImagePygletWingow
from RL.common.wrappers import MaxEpisodeStepsWrapper
from RL.dqn.dqn import DQNAgent, DQNSensitivityVisualizerAgent  # noqa: F401
from RL.common.utils import TfRunningStats, dense_net, auto_conv_dense_net, tf_training_step, tf_scale, tf_safe_softmax


class RNDRewardSystem:
    class Scopes(Enum):
        states_placeholer = "states"
        states_rms = "states_rms"
        target_network = "target_network"
        prediction_network = "prediction_network"
        error = "prediction_error"

    def __init__(self, context: Context, name):
        self.context = context
        self.name = name
        with tf.variable_scope(self.name):
            self._setup_model()

    def _setup_model(self):
        self._states_placeholder = self._tf_states_placeholder(RNDRewardSystem.Scopes.states_placeholer.value)
        self._states_rms = TfRunningStats(list(self.context.env.observation_space.shape), RNDRewardSystem.Scopes.states_rms.value)
        self._states_normalized = self._states_rms.normalize(self._states_placeholder)

    def _setup_network(self, states, scope):
        with tf.variable_scope(scope):
            return auto_conv_dense_net(self.context.need_conv_net, states, self.context.convs, self.context.states_embedding_hidden_layers, self.context.activation_fn, [self.context.rnd_num_features], )



# class RNDAgent(DQNAgent):
#     def __init__(self, context: Context, name):
#         super().__init__(context, name)
        