from collections import deque

import gym
import numpy as np
from gym.wrappers import AtariPreprocessing, FrameStack, TimeLimit

import RL
from RL.common.atari_wrappers import (ClipRewardEnv, EpisodicLifeEnv,
                                      FireResetEnv, NoopResetEnv)
from RL.common.utils import ImagePygletWingow, need_conv_net


class DummyWrapper(gym.Wrapper):
    def __init__(self, env):
        RL.logger.log("Wrapping with", str(type(self)))
        super().__init__(env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class DummyWrapper2(gym.Wrapper):
    def __init__(self, env):
        RL.logger.log("Wrapping with", str(type(self)))
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
        self.frames = deque([], maxlen=self.k)
        space = env.observation_space  # type: gym.spaces.Box
        assert len(space.shape) == 1  # can only stack 1-D frames
        self.observation_space = gym.spaces.Box(
            low=np.array(list(space.low) * self.k), high=np.array(list(space.high) * self.k))

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


MaxEpisodeStepsWrapper = TimeLimit


class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        r_total = 0
        for count in range(self.skip):
            obs, r, d, info = self.env.step(action)
            r_total += r
            if d:
                break
        return obs, r_total, d, info


class RenderWrapper(gym.Wrapper):
    def __init__(self, env, render_interval=1, vsync=True, caption="Renderer"):
        super().__init__(env)
        self.window = None
        try:
            self.window = ImagePygletWingow(caption=caption, vsync=vsync)
        except Exception as e:
            RL.logger.error(
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


class CostInfoWrapper(gym.Wrapper):
    def __init__(self, env, cost_extractor_fn):
        super().__init__(env)
        self.cost_extractor_fn = cost_extractor_fn

    def step(self, action):
        o, r, d, i = self.env.step(action)
        i['cost'] = i.get('cost', 0) + self.cost_extractor_fn(o, r, d, i)
        return o, r, d, i


def wrap_standard(env: gym.Env, c: RL.Context):
    env = CostInfoWrapper(env, lambda o, r, d, i: -i.get('Safety_reward', 0))  # for compatibility with older envs
    if need_conv_net(env.observation_space):
        env = FireResetEnv(env)
        env = AtariPreprocessing(env, c.atari_noop_max, c.atari_frameskip_k, terminal_on_life_loss=c.atari_episode_life)
        env = FrameStack(env, c.atari_framestack_k)
        if c.atari_clip_rewards:
            env = ClipRewardEnv(env)
        c.frameskip = c.atari_frameskip_k
    elif '-ram' in c.env_id:  # for playing atari from ram
        if c.atari_episode_life:
            env = EpisodicLifeEnv(env)
        env = NoopResetEnv(env, noop_max=c.atari_noop_max)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
            RL.logger.log("Fire reset being used")
        env = FrameSkipWrapper(env, skip=c.atari_frameskip_k)
        if c.atari_clip_rewards:
            env = ClipRewardEnv(env)
        c.frameskip = c.atari_frameskip_k
    else:
        if c.frameskip > 1:
            env = FrameSkipWrapper(env, skip=c.frameskip)
        # TODO: Add Framestack here:
    if c.artificial_max_episode_steps:
        env = TimeLimit(env, max_episode_steps=c.artificial_max_episode_steps)
    return env
