from RL.envs.mujoco.gather.gather_env import GatherEnv
from RL.envs.mujoco.ant_env import AntEnv
import gym


class AntGatherEnv(GatherEnv, gym.Env):

    MODEL_CLASS = AntEnv
    ORI_IND = 6
