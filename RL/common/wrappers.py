from collections import deque

import gym
import numpy as np

from RL.common import logger
from RL.common.utils import ImagePygletWingow


class DummyWrapper(gym.Wrapper):
    def __init__(self, env):
        logger.log("Wrapping with", str(type(self)))
        super().__init__(env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class DummyWrapper2(gym.Wrapper):
    def __init__(self, env):
        logger.log("Wrapping with", str(type(self)))
        super().__init__(env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class CartPoleWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1., 1., shape=[1])

    def step(self, action):
        if action[0] < 0:
            a = 0
        else:
            a = 1
        return super().step(a)


class ActionSpaceNormalizeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._ac_low = self.action_space.low
        self._ac_high = self.action_space.high
        self.action_space = gym.spaces.Box(
            -1, 1, shape=self.env.action_space.shape, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1, 1)
        action_correct = self._ac_low + \
            (self._ac_high - self._ac_low) * (action + 1) / 2
        return super().step(action_correct)


class LinearFrameStackWrapper(gym.Wrapper):
    k = 3

    def __init__(self, env, k=None):
        super().__init__(env)
        if k is not None:
            self.k = k
        else:
            self.k = LinearFrameStackWrapper.k
        self.frames = deque([], maxlen=k)
        space = env.observation_space  # type: gym.spaces.Box
        assert len(space.shape) == 1  # can only stack 1-D frames
        self.observation_space = gym.spaces.Box(
            low=np.array(list(space.low) * k), high=np.array(list(space.high) * k))

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._observation()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        obs = np.concatenate(self.frames, axis=0)
        assert list(np.shape(obs)) == list(self.observation_space.shape)
        return obs


class BipedalWrapper(gym.Wrapper):
    max_episode_length = 400

    def __init__(self, env):
        super().__init__(env)
        self.frame_count = 0

    def reset(self):
        self.frame_count = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frame_count += 1
        if self.frame_count >= self.max_episode_length:
            # reward -= 100
            done = True
        return obs, reward, done, info


class DiscreteToContinousWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(0, 1, shape=[env.action_space.n])

    def step(self, action):
        a = np.argmax(action)
        return super().step(a)


class MaxEpisodeStepsWrapper(gym.Wrapper):
    def __init__(self, env, max_steps=1000):
        super().__init__(env)
        self.max_steps = max_steps
        self._current_ep_steps = 0

    def reset(self):
        self._current_ep_steps = 0
        return self.env.reset()

    def step(self, action):
        obs, r, d, info = self.env.step(action)
        self._current_ep_steps += 1
        if self._current_ep_steps >= self.max_steps:
            d = True
        return obs, r, d, info


class RenderWrapper(gym.Wrapper):
    def __init__(self, env, render_interval=1, vsync=True, caption="Renderer"):
        super().__init__(env)
        self.window = None
        try:
            self.window = ImagePygletWingow(caption=caption, vsync=vsync)
        except Exception as e:
            logger.error(
                "RenderWrapper: Could not create window. Reason = {0}".format(str(e)))
        self.ep_id = 0
        self.render_interval = render_interval

    def draw(self, obs):
        if self.window and self.ep_id % self.render_interval == 0:
            self.window.imshow(self.env.render('rgb_array'))

    def reset(self):
        obs = self.env.reset()
        self.draw(obs)
        return obs

    def step(self, action):
        obs_next, r, d, info = self.env.step(action)
        self.draw(obs_next)
        if d:
            self.ep_id += 1
        return obs_next, r, d, info

    def close(self):
        if self.window:
            self.window.close()
