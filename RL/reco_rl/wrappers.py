import os
from collections import deque

import gym
import numpy as np
from gym import error

from RL.common import logger
from RL.common.utils import scale
from RL.reco_rl.constraints import (convert_to_constraints_dict,
                                    count_leaf_nodes_in_constraints,
                                    cplex_nearest_feasible,
                                    normalize_constraints)


class ERSEnvWrapper(gym.Wrapper):
    k = 3

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.k = ERSEnvWrapper.k
        self.request_heat_maps = deque([], maxlen=self.k)
        self.n_ambs = self.metadata['nambs']
        self.n_bases = env.action_space.shape[0]
        self.action_space = gym.spaces.Box(0, 1, shape=[self.n_bases])
        self.observation_space = gym.spaces.Box(
            0, 1, shape=[self.k * self.n_bases + self.n_bases + 1])

    def compute_alloc(self, action):
        action = np.clip(action, 0, 1)
        remaining = 1
        alloc = np.zeros([self.n_bases])
        for i in range(len(action)):
            alloc[i] = action[i] * remaining
            remaining -= alloc[i]
        alloc[-1] = remaining
        assert all(alloc >= 0) and all(
            alloc <= 1), "alloc is {0}".format(alloc)
        # assert sum(alloc) == 1, "sum is {0}".format(sum(alloc))
        return alloc

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs = self.env.reset()
        for _ in range(self.k):
            self.request_heat_maps.append(self.obs[0:self.n_bases])
        return self._observation()

    def step(self, action):
        # action = self.compute_alloc(action)
        logger.log('alloc: {0}'.format(
            np.round(action * self.n_ambs, 2)), level=logger.DEBUG)
        self.obs, r, d, _ = super().step(action)
        self.request_heat_maps.append(self.obs[0:self.n_bases])
        return self._observation(), r, d, _

    def _observation(self):
        assert len(self.request_heat_maps) == self.k
        obs = np.concatenate((np.concatenate(
            self.request_heat_maps, axis=0), self.obs[self.n_bases:]), axis=0)
        if logger.Logger.CURRENT.level <= logger.DEBUG:
            logger.log('req_heat_map: {0}'.format(
                np.round(self.obs[0:self.n_bases], 2)), level=logger.DEBUG)
        return obs


class ERSEnvImWrapper(gym.Wrapper):
    k = 3

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.k = ERSEnvImWrapper.k
        self.request_heat_maps = deque([], maxlen=self.k)
        self.n_ambs = self.metadata['nambs']
        self.n_bases = env.action_space.shape[0]
        self.action_space = gym.spaces.Box(0, 1, shape=[self.n_bases])
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shp[0], shp[1], self.k + shp[2] - 1))

    def compute_alloc(self, action):
        action = np.clip(action, 0, 1)
        remaining = 1
        alloc = np.zeros([self.n_bases])
        for i in range(len(action)):
            alloc[i] = action[i] * remaining
            remaining -= alloc[i]
        alloc[-1] = remaining
        assert all(alloc >= 0) and all(
            alloc <= 1), "alloc is {0}".format(alloc)
        # assert sum(alloc) == 1, "sum is {0}".format(sum(alloc))
        return alloc

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs = self.env.reset()
        for _ in range(self.k):
            self.request_heat_maps.append(self.obs[:, :, 0:1])
        return self._observation()

    def step(self, action):
        # action = self.compute_alloc(action)
        logger.log('alloc: {0}'.format(
            np.round(action * self.n_ambs, 2)), level=logger.DEBUG)
        self.obs, r, d, _ = super().step(action)
        self.request_heat_maps.append(self.obs[:, :, 0:1])
        return self._observation(), r, d, _

    def _observation(self):
        assert len(self.request_heat_maps) == self.k
        obs = np.concatenate((np.concatenate(
            self.request_heat_maps, axis=2), self.obs[:, :, 1:]), axis=2)
        if logger.Logger.CURRENT.level <= logger.DEBUG:
            logger.log('req_heat_map: {0}'.format(
                np.round(self.obs[:, :, 0], 2)), level=logger.DEBUG)
        assert list(obs.shape) == [21, 21, 5]
        return obs


class ERStoMMDPWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        logger.log("Wrapping with", str(type(self)))
        super().__init__(env)
        self.metadata['nzones'] = self.metadata['nbases']
        self.metadata['nresources'] = self.metadata['nambs']
        if 'constraints' not in self.metadata or self.metadata['constraints'] is None:
            self.metadata['constraints'] = convert_to_constraints_dict(
                self.metadata['nzones'], self.metadata['nresources'], env.action_space.low, env.action_space.high)
        assert count_leaf_nodes_in_constraints(
            self.metadata['constraints']) == self.metadata['nzones'], "num of leaf nodes in constraints tree should be same as number of zones"

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class BSStoMMDPWrapper(gym.Wrapper):
    def __init__(self, env):
        logger.log("Wrapping with", str(type(self)))
        super().__init__(env)
        self.metadata['nzones'] = self.metadata['nzones']
        self.metadata['nresources'] = self.metadata['nbikes']
        if 'constraints' not in self.metadata or self.metadata['constraints'] is None:
            self.metadata['constraints'] = convert_to_constraints_dict(
                self.metadata['nzones'], self.metadata['nresources'], env.action_space.low, env.action_space.high)
        assert count_leaf_nodes_in_constraints(
            self.metadata['constraints']) == self.metadata['nzones'], "num of leaf nodes in constraints tree should be same as number of zones"

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class MMDPActionSpaceNormalizerWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        logger.log("Wrapping with", str(type(self)))
        super().__init__(env)
        self.nzones = self.metadata['nzones']
        self.nresources = self.metadata['nresources']
        ac_space_low = env.action_space.low / self.nresources
        ac_space_high = env.action_space.high / self.nresources
        assert np.all(
            ac_space_low >= 0), 'Invalid action space. Individual requirements must be greater than or equal to 0'
        assert np.all(
            ac_space_high <= 1), 'Invalid action space. Individual capacities must be less than or equal the global capacity'
        assert np.sum(
            ac_space_low) < 1, 'Invalid action space. Sum of individual requirements must be less than total resources'
        assert np.sum(
            ac_space_high) > 1, 'Invalid action space. Sum of individual capacities must be greater than total resources'
        self.action_space = gym.spaces.Box(
            low=ac_space_low, high=ac_space_high, dtype=np.float32)
        logger.log('ac space low: ', str(self.action_space.low))
        logger.log('ac space high: ', str(self.action_space.high))
        normalize_constraints(self.metadata["constraints"])

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def step(self, action):
        # action = [1 / self.nzones] * self.nzones
        allocation_fraction = action * self.nresources
        allocation = np.round(allocation_fraction)
        if np.sum(allocation) != self.nresources:
            raise error.InvalidAction(
                "Invalid action. The action, when rounded, should sum to nresources. Provided action was {0}".format(allocation))
        logger.log("action: {0}".format(allocation), level=logger.DEBUG)
        self.obs, r, d, info = self.env.step(allocation)
        return self.obs, r, d, info


class MMDPActionWrapper(gym.Wrapper):
    """Must wrap MMDPActionSpaceNormalizerWrapper. i.e. assumes action space is already normalized"""

    def __init__(self, env: gym.Env, assume_feasible_action_input=False):
        logger.log("Wrapping with", str(type(self)))
        super().__init__(env)
        self.assume_feasible_action_input = assume_feasible_action_input
        if self.assume_feasible_action_input:
            logger.log("Assuming feasible action will be given to action wrapper")
        else:
            logger.log("Action wrapper will feasiblize given action using a QP")
        logger.log("Action wrapper will round off the action")
        self.nresources = self.env.metadata['nresources']
        self._logged_cplex_problem_yet = False

    def _get_allocation(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        if abs(sum(action) - 1) > 1e-6:
            raise error.InvalidAction(
                "Invalid action. The action must sum to 1. Provided action was {0}".format(action))
        if np.any(action < -0.0):
            raise ValueError(
                "Each dimension of action must be >=0. Provided action was {0}".format(action))
        if np.any(action > 1.0):
            raise ValueError(
                "Each dimension of action must be <=1. Provided action was {0}".format(action))

        allocation_fraction = action * self.nresources
        allocation = np.round(allocation_fraction)
        # print(allocation)
        allocated = np.sum(allocation)
        deficit_per_zone = allocation_fraction - allocation
        deficit = self.nresources - allocated
        # print('deficit: {0}'.format(deficit))
        while deficit != 0:
            increase = int(deficit > 0) - int(deficit < 0)
            # print('increase: {0}'.format(increase))
            target_zone = np.argmax(increase * deficit_per_zone)
            # print('target zone: {0}'.format(target_zone))
            allocation[target_zone] += increase
            # print('alloction: {0}'.format(allocation))
            allocated += increase
            deficit_per_zone[target_zone] -= increase
            deficit -= increase
            # print('deficit: {0}'.format(deficit))
        return allocation

    def round_action(self, action):
        '''assumes action is feasible'''
        action = self._get_allocation(action) / self.nresources
        return action

    def wrap(self, action):
        if not self.assume_feasible_action_input:
            solution = cplex_nearest_feasible(action.astype(
                np.float64), self.env.metadata['constraints'])
            feasible_action = solution['feasible_action']
            if not self._logged_cplex_problem_yet:
                solution['prob'].write(os.path.join(
                    logger.get_dir(), 'nearest_feasible_cplex.lp'))
                self._logged_cplex_problem_yet = True
        else:
            feasible_action = action
        rounded_action = self.round_action(feasible_action)
        change = np.mean(np.abs(rounded_action - action))
        return rounded_action, change

    def reset(self):
        return self.env.reset()

    def step(self, action):
        action, change = self.wrap(action)
        return self.env.step(action)


class MMDPObsNormalizeWrapper(gym.Wrapper):
    """Must be used before MMDPObsStackWrapper"""

    def __init__(self, env):
        logger.log("Wrapping with", str(type(self)))
        super().__init__(env)
        self.nzones = self.env.metadata['nzones']
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=self.env.observation_space.shape, dtype=np.float32)
        logger.log('ob space low: ', str(self.observation_space.low))
        logger.log('ob space high: ', str(self.observation_space.high))

    def _transform_obs(self, obs):
        logger.log('demand: ', str(obs[0:self.nzones]), level=logger.DEBUG)
        logger.log('cur_alloc: ', str(
            obs[self.nzones: 2 * self.nzones]), level=logger.DEBUG)
        logger.log('cur_time: ', str(obs[-1]), level=logger.DEBUG)
        return scale(obs, self.env.observation_space.low, self.env.observation_space.high, 0, 1)

    def reset(self):
        return self._transform_obs(self.env.reset())

    def step(self, action):
        obs, r, d, info = self.env.step(action)
        return self._transform_obs(obs), r, d, info


class MMDPObsStackWrapper(gym.Wrapper):
    k = 3

    def __init__(self, env):
        logger.log("Wrapping with", str(type(self)))
        super().__init__(env)
        self.last_k_demands = deque([], maxlen=self.k)
        self.nzones = self.metadata['nzones']
        low = list(self.env.observation_space.low)
        low = low[0:self.nzones] * self.k + low[self.nzones:]
        high = list(self.env.observation_space.high)
        high = high[0:self.nzones] * self.k + high[self.nzones:]
        self.observation_space = gym.spaces.Box(
            low=np.array(low), high=np.array(high), dtype=self.env.observation_space.dtype)

    def _observation(self):
        assert len(self.last_k_demands) == self.k
        obs = np.concatenate((np.concatenate(
            self.last_k_demands, axis=0), self.obs[self.nzones:]), axis=0)
        return obs

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs = self.env.reset()
        for _ in range(self.k):
            self.last_k_demands.append(self.obs[0:self.nzones])
        return self._observation()

    def step(self, action):
        self.obs, r, d, info = self.env.step(action)
        self.last_k_demands.append(self.obs[0:self.nzones])
        return self._observation(), r, d, info
