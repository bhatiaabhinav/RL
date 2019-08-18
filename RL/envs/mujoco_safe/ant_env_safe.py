from rllab.core.serializable import Serializable
from RL.envs.mujoco.ant_env import AntEnv
from RL.envs.mujoco_safe.mujoco_env_safe import SafeMujocoEnv
import gym


class SafeAntEnv(SafeMujocoEnv, Serializable, gym.Env):

    MODEL_CLASS = AntEnv

    def render(self, mode='human'):
        super().render(mode=mode)
