from RL.envs.mujoco.gather.gather_env import GatherEnv
from RL.envs.mujoco.point_env import PointEnv
import gym


class PointGatherEnv(GatherEnv, gym.Env):

    MODEL_CLASS = PointEnv
    ORI_IND = 2
