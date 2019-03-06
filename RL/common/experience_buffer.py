import sys

import numpy as np
from typing import List
from RL.common import logger


class Experience:
    def __init__(self, state, action, reward, done, info, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.info = info
        self.next_state = next_state

    def __sizeof__(self):
        return sys.getsizeof(self.state) + sys.getsizeof(self.action) + \
            sys.getsizeof(self.reward) + sys.getsizeof(self.done) + \
            sys.getsizeof(self.info) + sys.getsizeof(self.next_state)


class MultiRewardStreamExperience(Experience):
    '''The only difference from 'Experience' is that 'reward' attribute is expected to be a numpy list'''
    def __init__(self, state, action, reward, done, info, next_state):
        assert(hasattr(reward, "__len__")), 'reward attribute should be a list'
        reward = np.asarray(reward)
        super().__init__(state, action, reward, done, info, next_state)


class ExperienceBuffer:
    '''A circular buffer to hold experiences'''
    def __init__(self, length=1e6, size_in_bytes=None):
        self.buffer = []  # type: List[Experience]
        self.buffer_length = length
        self.count = 0
        self.size_in_bytes = size_in_bytes
        self.next_index = 0

    def __len__(self):
        return self.count

    def add(self, exp: Experience):
        if self.count == 0:
            if self.size_in_bytes is not None:
                self.buffer_length = self.size_in_bytes / sys.getsizeof(exp)
            self.buffer_length = int(self.buffer_length)
            logger.log('Initializing experience buffer of length {0}. Est. memory: {1} MB'.format(
                self.buffer_length, self.buffer_length * sys.getsizeof(exp) // (1024 * 1024)))
            self.buffer = [None] * self.buffer_length
        self.buffer[self.next_index] = exp
        self.next_index = (self.next_index + 1) % self.buffer_length
        self.count = min(self.count + 1, self.buffer_length)

    def random_experiences(self, count):
        indices = np.random.randint(0, self.count, size=count)
        for i in indices:
            yield self.buffer[i]

    def random_experiences_unzipped(self, count):
        exps = self.random_experiences(count)
        list_of_exp_tuples = [(exp.state, exp.action, exp.reward, exp.done, exp.info, exp.next_state) for exp in exps]
        return tuple(np.asarray(tup) for tup in zip(*list_of_exp_tuples))

    def random_rollouts(self, count, rollout_size):
        starting_indices = np.random.randint(0, self.count - rollout_size, size=count)
        rollouts = []
        for i in starting_indices:
            rollout = self.buffer[i:i + rollout_size]
            rollouts.append(rollout)
        return np.array(rollouts)

    def random_rollouts_unzipped(self, count, rollout_size, dones_as_ints=True):
        starting_indices = np.random.randint(0, self.count - rollout_size, size=count)
        states, actions, rewards, dones, infos, next_states = [], [], [], [], [], []
        for i in starting_indices:
            rollout = self.buffer[i:i + rollout_size]
            states.append([exp.state for exp in rollout])
            actions.append([exp.action for exp in rollout])
            rewards.append([exp.reward for exp in rollout])
            dones.append([int(exp.done) if dones_as_ints else exp.done for exp in rollout])
            infos.append([exp.info for exp in rollout])
            next_states.append([exp.next_state for exp in rollout])
        states, actions, rewards, dones, infos, next_states = np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(infos), np.array(next_states)
        for item in [states, actions, rewards, dones, infos, next_states]:
            assert list(item.shape[0:2]) == [count, rollout_size], "item: {0}, shape: {1}, expected: {2}".format(item, list(item.shape), [count, rollout_size])
        return states, actions, rewards, dones, infos, next_states

    def random_states(self, count):
        experiences = list(self.random_experiences(count))
        return [e.state for e in experiences]
