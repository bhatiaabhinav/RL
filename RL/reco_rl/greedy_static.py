import json
import logging
import os.path as osp
import time

import gym
import numpy as np

import gym_ERSLE  # noqa: F401
from RL.common import logger
from RL.common.utils import set_global_seeds
from RL.common.vec_env.subproc_vec_env import SubprocVecEnv


def reseed(env, seed):
    seeds = [seed + i for i in range(env.num_envs)]
    print('(Re)seeding the envs with seeds {0}'.format(seeds))
    env.seed(seeds)


def get_actions(num_envs, action_id):
    return [action_id for i in range(num_envs)]


def find_num_ambs(info):
    num_ambs = 0
    for key in info.keys():
        if key.startswith('ambs'):
            num_ambs += info[key]
    return num_ambs


def find_num_bases(info):
    num_bases = 0
    for key in info.keys():
        if key.startswith('base'):
            num_bases += 1
    return num_bases


def allocate(env, target_allocation, num_ambs, current_allocation=None):
    if sum(target_allocation) != num_ambs:
        raise ValueError('sum of target_allocation should be same as num_ambs')
    num_bases = len(target_allocation)
    av_reward = 0

    if current_allocation is None:
        # figure out the current_allocation:
        obs, r, dones, info = env.step(get_actions(env.num_envs, 0))
        av_reward += np.average(r)
        current_allocation = []
        for base in range(num_bases):
            current_allocation.append(info[0]['base' + str(base)])
    assert(sum(current_allocation) == num_ambs)

    change = (np.asarray(target_allocation) -
              np.asarray(current_allocation)).tolist()
    assert(sum(change) == 0)
    change = [{'base': i, 'change': change[i]} for i in range(num_bases)]
    largest_sources = sorted(change, key=lambda item: item['change'])
    largest_gainers = sorted(
        change, key=lambda item: item['change'], reverse=True)
    ls_index = 0
    lg_index = 0
    while max(ls_index, lg_index) < num_bases:
        while largest_sources[ls_index]['change'] < 0 and largest_gainers[lg_index]['change'] > 0:
            src = largest_sources[ls_index]['base']
            dst = largest_gainers[lg_index]['base']
            action = 1 + dst * (num_bases - 1) + \
                src if src < dst else dst * (num_bases - 1) + src
            obs, r, dones, info = env.step(get_actions(env.num_envs, action))
            av_reward += np.average(r)
            largest_sources[ls_index]['change'] += 1
            largest_gainers[lg_index]['change'] -= 1
        if largest_sources[ls_index]['change'] == 0:
            ls_index += 1
        if largest_gainers[lg_index]['change'] == 0:
            lg_index += 1

    for c in change:
        assert(c['change'] == 0)

    return av_reward


def allocate_per_amb(env, allocation_per_amb, num_ambs, num_bases):
    # place all ambs on base 0:
    av_reward = 0
    for src_base in range(1, num_bases):
        base_has_amb = True
        while base_has_amb:
            obs, r, dones, info = env.step(get_actions(env.num_envs, src_base))
            base_has_amb = info[0]['base' + str(src_base)] > 0
            av_reward += np.average(r)
    # now place according to given alloc:
    for dest_base in allocation_per_amb:
        src_base = 0
        # action_id = dest_base * num_bases + src_base
        action_id = 1 + dest_base * \
            (num_bases - 1) + src_base if src_base < dest_base else dest_base * \
            (num_bases - 1) + src_base
        obs, r, dones, info = env.step(get_actions(env.num_envs, action_id))
        av_reward += np.average(r)
    return av_reward


def test_candidate(allocation_per_base_so_far, candidate_base, env, num_ambs, num_bases, seed):
    reseed(env, seed)
    env.reset()
    av_reward = 0

    # calculate target allocation and allocate
    target_allocation = allocation_per_base_so_far.copy()
    target_allocation[candidate_base] += 1
    # rest go to base 0
    target_allocation[0] += num_ambs - sum(target_allocation)
    # target_allocation = np.array(target_allocation) / np.sum(target_allocation)
    av_reward = 0
    done = False
    while not done:
        obs, r, dones, info = env.step(
            get_actions(env.num_envs, target_allocation))
        done = dones[0]
        av_reward += np.average(r)
    return av_reward


def optimize(policy, env, seed, ob_dtype='uint8', nsteps=5, nstack=4, total_timesteps=int(80e6), frameskip=1, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, _lambda=1.0, log_interval=100, saved_model_path=None, render=False, no_training=False):
    set_global_seeds(seed)
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    if logger.get_dir():
        with open(osp.join(logger.get_dir(), 'params.json'), 'w') as f:
            f.write(json.dumps({'policy': str(policy), 'env_id': env.id, 'nenvs': nenvs, 'seed': seed, 'ac_space': str(ac_space), 'ob_space': str(ob_space), 'ob_type': ob_dtype, 'nsteps': nsteps, 'nstack': nstack, 'total_timesteps': total_timesteps, 'frameskip': frameskip, 'vf_coef': vf_coef, 'ent_coef': ent_coef,
                                'max_grad_norm': max_grad_norm, 'lr': lr, 'lrschedule': lrschedule, 'epsilon': epsilon, 'alpha': alpha, 'gamma': gamma, 'lambda': _lambda, 'log_interval': log_interval, 'saved_model_path': saved_model_path, 'render': render, 'no_training': no_training, 'abstime': time.time()}))

    version_to_amb_map = gym_ERSLE.version_to_ambs_map
    num_ambs = version_to_amb_map['v' + env.id[-1]]
    num_bases = ac_space.shape[0]
    print('Num ambs: {0}\tNum bases: {1}\n'.format(num_ambs, num_bases))

    reward_so_far = 0
    allocation_per_amb_so_far = []
    allocation_per_base_so_far = [0] * num_bases

    # then one by one, place ambulances according to maximum gain
    for amb in range(num_ambs):
        print('Finding best base for amb {0}. Allocation for prev ambs: {1}'.format(
            amb, allocation_per_amb_so_far))
        print('-------------------------------------------------------------------------------------------')
        best_base = 0
        best_gain = -1000000
        for candidate_base in range(num_bases):
            gain = test_candidate(allocation_per_base_so_far, candidate_base,
                                  env, num_ambs, num_bases, seed) - reward_so_far
            if gain > best_gain:
                best_gain = gain
                best_base = candidate_base
                print('Tried base {0}:\tGain: {1}\t[Best so far]'.format(
                    candidate_base, gain))
            else:
                print('Tried base {0}:\tGain: {1}'.format(
                    candidate_base, gain))

        allocation_per_amb_so_far.append(best_base)
        allocation_per_base_so_far[best_base] += 1
        reward_so_far += best_gain
        print('Best base for amb {0} is {1}'.format(amb, best_base))
        print('Best score so far: {0}'.format(reward_so_far))
        print('Base wise alloction so far: {0}'.format(
            allocation_per_base_so_far))
        print('-------------------------------------')
        print('-------------------------------------')
        print('')

    print('Final Allocation')
    print('-----------------')
    print('Ambulance wise: {0}'.format(allocation_per_amb_so_far))
    print('Base wise: {0}'.format(allocation_per_base_so_far))
    print('Score: {0}'.format(reward_so_far))
    print('-------------------------------------')
    print('-------------------------------------')
    print('')

    # print('Now just playing ... :)')
    # print('Do not expect same scores because the envs will not be reseeded now on.\n')
    # while True:
    #     av_reward = 0
    #     av_reward += allocate(env, allocation_per_base_so_far, num_ambs)
    #     done = False
    #     while not done:
    #         obs, r, dones, info = env.step(get_actions(env.num_envs, 0))
    #         av_reward += np.average(r)
    #         done = dones[0]
    #     print('Score: {0}'.format(av_reward))


def main():
    from baselines.ers.args import parse
    args = parse()

    def make_env(rank):
        def _thunk():
            env = gym.make(args.env)
            env.seed(args.seed + rank)
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk
    set_global_seeds(args.seed)
    env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])
    env.id = args.env
    optimize("greedy", env, args.seed, args.ob_dtype, render=args.render)


if __name__ == '__main__':
    main()
