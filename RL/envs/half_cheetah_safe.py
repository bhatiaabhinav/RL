from gym.envs.mujoco import HalfCheetahEnv
import numpy as np


class HalfCheetahSafeEnv(HalfCheetahEnv):
    def __init__(self, max_speed=1.0, horizon=np.inf):
        self.max_speed = max_speed
        self.horizon = horizon
        self.time = 0
        super().__init__()

    def _get_obs(self):
        obs = super()._get_obs()
        if self.horizon < np.inf:
            obs = np.concatenate((obs, [1 - self.time / self.horizon]))
        return obs

    def reset(self):
        obs = super().reset()
        self.time = 0
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.time += 1
        vx = info['reward_run']
        info['Safety_reward'] = -float(abs(vx) > self.max_speed)
        done = done or self.time >= self.horizon
        return obs, reward, done, info
