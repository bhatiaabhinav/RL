import sys

import numpy as np
from typing import List
import RL

ids = 0


class Experience:
    def __init__(self, state, action, reward, done, info, next_state, cost=0):
        self.state = state
        self.action = action
        self.reward = reward
        self.cost = cost
        self.done = done
        self.info = info
        self.next_state = next_state
        self.id = 0

    def __sizeof__(self):
        return sys.getsizeof(self.state) + sys.getsizeof(self.action) + \
            sys.getsizeof(self.reward) + sys.getsizeof(self.done) + \
            sys.getsizeof(self.info) + sys.getsizeof(self.next_state) + \
            sys.getsizeof(self.cost)


# class MultiRewardStreamExperience(Experience):
#     '''The only difference from 'Experience' is that 'reward' attribute is expected to be a numpy list'''
#     def __init__(self, state, action, reward, done, info, next_state):
#         RL.logger.warn("Multiple rewards might not work anymore. Change the code.")
#         assert(hasattr(reward, "__len__")), 'reward attribute should be a list'
#         reward = np.asarray(reward)
#         super().__init__(state, action, reward, done, info, next_state)


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
            RL.logger.debug('Initializing experience buffer of length {0}. Est. memory: {1} MB'.format(
                self.buffer_length, self.buffer_length * sys.getsizeof(exp) // (1024 * 1024)))
            self.buffer = [None] * self.buffer_length
        self.buffer[self.next_index] = exp
        self.next_index = (self.next_index + 1) % self.buffer_length
        self.count = min(self.count + 1, self.buffer_length)
        global ids
        self.id = ids
        ids += 1

    def random_experiences(self, count):
        indices = np.random.randint(0, self.count, size=count)
        # if self.count < self.buffer_length:
        #     priors = np.asarray([1 / ((self.count - i) ** 0.5) for i in range(self.count)])
        #     priors = priors / np.sum(priors)
        #     indices = np.random.choice(self.count, size=count, p=priors)
        for i in indices:
            yield self.buffer[i]

    def random_experiences_unzipped(self, count, return_costs=False):
        exps = self.random_experiences(count)
        if return_costs:
            list_of_exp_tuples = [(exp.state, exp.action, exp.reward, exp.cost, exp.done, exp.info, exp.next_state) for exp in exps]
        else:
            list_of_exp_tuples = [(exp.state, exp.action, exp.reward, exp.done, exp.info, exp.next_state) for exp in exps]
        return tuple(np.asarray(tup) for tup in zip(*list_of_exp_tuples))

    def random_rollouts(self, count, rollout_size):
        starting_indices = np.random.randint(0, self.count - rollout_size, size=count)
        rollouts = []
        for i in starting_indices:
            rollout = self.buffer[i:i + rollout_size]
            rollouts.append(rollout)
        return np.array(rollouts)

    def random_rollouts_unzipped(self, count, rollout_size, dones_as_ints=True, return_costs=False):
        starting_indices = np.random.randint(0, self.count - rollout_size, size=count)
        states, actions, rewards, dones, infos, next_states = [], [], [], [], [], []
        if return_costs:
            costs = []
        for i in starting_indices:
            rollout = self.buffer[i:i + rollout_size]
            states.append([exp.state for exp in rollout])
            actions.append([exp.action for exp in rollout])
            rewards.append([exp.reward for exp in rollout])
            dones.append([int(exp.done) if dones_as_ints else exp.done for exp in rollout])
            infos.append([exp.info for exp in rollout])
            next_states.append([exp.next_state for exp in rollout])
            if return_costs:
                costs.append([exp.cost for exp in rollout])
        states, actions, rewards, dones, infos, next_states = np.asarray(states), np.asarray(actions), np.asarray(rewards), np.asarray(dones), np.asarray(infos), np.asarray(next_states)
        if return_costs:
            costs = np.asarray(costs)
        if return_costs:
            return_items = (states, actions, rewards, costs, dones, infos, next_states)
        else:
            return_items = (states, actions, rewards, dones, infos, next_states)
        for item in return_items:
            assert list(item.shape[0:2]) == [count, rollout_size], "item: {0}, shape: {1}, expected: {2}".format(item, list(item.shape), [count, rollout_size])
        return return_items

    def random_states(self, count):
        experiences = list(self.random_experiences(count))
        return np.asarray([e.state for e in experiences])
