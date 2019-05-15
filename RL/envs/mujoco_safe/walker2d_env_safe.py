from rllab.core.serializable import Serializable
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from RL.envs.mujoco_safe.mujoco_env_safe import SafeMujocoEnv
import gym


class SafeWalker2DEnv(SafeMujocoEnv, Serializable, gym.Env):

    MODEL_CLASS = Walker2DEnv
