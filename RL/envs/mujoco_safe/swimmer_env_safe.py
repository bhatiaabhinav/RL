from rllab.core.serializable import Serializable
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from RL.envs.mujoco_safe.mujoco_env_safe import SafeMujocoEnv
import numpy as np
import gym


class SafeSwimmerEnv(SafeMujocoEnv, Serializable, gym.Env):

    MODEL_CLASS = SwimmerEnv

