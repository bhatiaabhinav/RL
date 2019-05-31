import gym
import numpy as np
from roboschool.gym_mujoco_walkers import RoboschoolForwardWalkerMujocoXML


class MyRS(RoboschoolForwardWalkerMujocoXML):
    '''
    2-D one-leg hopping robot similar to MuJoCo Hopper.
    The task is to make the hopper hop as fast as possible.
    '''
    foot_list = ["torso"]

    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "point.xml", "torso", action_dim=3, obs_dim=8, power=0.75)
        self.action_space = gym.spaces.Box(low=np.array([-1, -0.25]), high=np.array([1, 0.25]))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9, ))

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1
