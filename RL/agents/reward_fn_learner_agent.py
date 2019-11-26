import RL
import tensorflow as tf
import numpy as np
from RL.common.utils import tf_inputs, auto_conv_dense_net, tf_training_step
from RL.agents import ExperienceBufferAgent

# class RewardFnLearnerAgent:
#     def __init__(self, context: RL.Context, name, exp_buffer_agent: ExperienceBufferAgent, len_reward=1):
#         self.context = context
#         self.name = name
#         self.len_reward = len_reward

#     def tf_reward_fn_approximator(self, name, reuse=tf.AUTO_REUSE):
#         with tf.variable_scope(name):
            