from rllab.core.serializable import Serializable
from RL.envs.mujoco.point_env import PointEnv
from RL.envs.mujoco_safe.mujoco_env_safe import SafeMujocoEnv
import gym


class SafePointEnv(SafeMujocoEnv, Serializable, gym.Env):

    MODEL_CLASS = PointEnv

    def render(self, mode='human'):
        super().render(mode=mode)
