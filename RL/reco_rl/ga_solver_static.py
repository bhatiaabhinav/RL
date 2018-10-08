import json
import os.path as osp
import random
import time
from typing import List

import numpy as np

from RL.common import logger
from RL.common.utils import set_global_seeds

rand = random.Random()
rand.seed(0)
np.random.seed(0)
n_ambs = 18
n_bases = 6
n_actions = n_bases * (n_bases - 1) + 1
NO_OP = 0
beam_size = 4


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


def allocate(env, target_allocation, current_allocation=None):
    if sum(target_allocation) != n_ambs:
        raise ValueError('sum of target_allocation should be same as num_ambs')
    av_reward = 0

    if current_allocation is None:
        # figure out the current_allocation:
        obs, r, dones, info = env.step(get_actions(env.num_envs, 0))
        av_reward += np.average(r)
        current_allocation = []
        for base in range(n_bases):
            current_allocation.append(info[0]['base' + str(base)])
    assert(sum(current_allocation) == n_ambs)

    change = (np.asarray(target_allocation) - np.asarray(current_allocation)).tolist()
    assert(sum(change) == 0)
    change = [{'base': i, 'change': change[i]} for i in range(n_bases)]
    largest_sources = sorted(change, key=lambda item: item['change'])
    largest_gainers = sorted(change, key=lambda item: item['change'], reverse=True)
    ls_index = 0
    lg_index = 0
    while max(ls_index, lg_index) < n_bases:
        while largest_sources[ls_index]['change'] < 0 and largest_gainers[lg_index]['change'] > 0:
            src = largest_sources[ls_index]['base']
            dst = largest_gainers[lg_index]['base']
            action = 1 + dst * (n_bases - 1) + src if src < dst else dst * (n_bases - 1) + src
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


def eval_alloc(env, alloc, seed=0):
    reseed(env, seed)
    env.reset()
    R = allocate(env, alloc)
    d = False
    while not d:
        obs, r, dones, _ = env.step(get_actions(env.num_envs, NO_OP))
        d = dones[0]
        R += np.average(r)
    return R


def mutated(alloc: List[int], max_mutations=4):
    a = alloc.copy()
    ambs = np.sum(alloc)
    for i in range(rand.randint(1, max_mutations)):
        src = rand.randint(0, len(alloc) - 1)
        dst = rand.randint(0, len(alloc) - 1)
        if a[src] > 0 and a[dst] < ambs:
            a[src] -= 1
            a[dst] += 1
    return a


def mutations(alloc, n=2, max_mutations=4):
    for i in range(n):
        yield mutated(alloc, max_mutations)


def normalize(alloc):
    sum = np.sum(alloc)
    if sum == 0:
        alloc = np.array([n_ambs // n_bases] * n_bases)
    else:
        alloc = (np.asarray(alloc) * n_ambs) // np.sum(alloc)
    sum = np.sum(alloc)
    if sum < n_ambs:
        deficit = n_ambs - sum
        alloc[0:deficit] += np.ones([deficit], dtype=np.int32)
    assert(np.sum(alloc) == n_ambs)
    return alloc


def crossover(alloc1, alloc2):
    alloc = alloc1.copy()
    for i in range(len(alloc)):
        if rand.random() > 0.5:
            alloc[i] = alloc2[i]
    return alloc


def random_alloc():
    alloc = [0] * n_bases
    remaining = n_ambs
    for b in range(n_bases - 1):
        alloc[b] = rand.randint(0, remaining)
        remaining -= alloc[b]
    alloc[n_bases - 1] = remaining
    return alloc


def optimize(policy, env, seed, generations=30, ob_dtype='uint8', nsteps=5, nstack=4, total_timesteps=int(80e6), frameskip=1, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, _lambda=1.0, log_interval=100, saved_model_path=None, render=False, no_training=False):
    set_global_seeds(seed)
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    if logger.get_dir():
        with open(osp.join(logger.get_dir(), 'params.json'), 'w') as f:
            f.write(json.dumps({'policy': str(policy), 'env_id': env.id, 'nenvs': nenvs, 'seed': seed, 'ac_space': str(ac_space), 'ob_space': str(ob_space), 'ob_type': ob_dtype, 'nsteps': nsteps, 'nstack': nstack, 'total_timesteps': total_timesteps, 'frameskip': frameskip, 'vf_coef': vf_coef, 'ent_coef': ent_coef,
                                'max_grad_norm': max_grad_norm, 'lr': lr, 'lrschedule': lrschedule, 'epsilon': epsilon, 'alpha': alpha, 'gamma': gamma, 'lambda': _lambda, 'log_interval': log_interval, 'saved_model_path': saved_model_path, 'render': render, 'no_training': no_training, 'abstime': time.time()}))

    obs = env.reset()
    obs, r, d, info = env.step(get_actions(nenvs, 0))
    num_ambs = find_num_ambs(info[0])
    num_bases = find_num_bases(info[0])
    print('Num ambs: {0}\tNum bases: {1}\n'.format(num_ambs, num_bases))

    population = [random_alloc() for i in range(beam_size)]
    scores = [eval_alloc(env, p) for p in population]
    # print(population)
    for g in range(generations):
        print('Generation {0}'.format(g))
        print('-------------------------')

        # make kids:
        population_new = []
        for p in population:
            # parent2 = population[random.randint(0, beam_size-1)]
            # population_new.append(mutated(normalize(crossover(p, parent2)), max_mutations=4))
            population_new.append(mutated(p, max_mutations=4))
        # print(population)

        # eval
        scores_new = np.array([eval_alloc(env, p) for p in population_new])

        population.extend(population_new)
        scores.extend(scores_new)

        # purge:
        p = np.array(scores)
        p -= np.min(p)
        p /= (np.max(p) / 3)
        p = np.exp(p)
        p /= np.sum(p)
        selected_indices = np.random.choice(len(population), size=beam_size, replace=False, p=p)
        population = list(np.asarray(population)[selected_indices])
        scores = list(np.asarray(scores)[selected_indices])

        # print stats:
        print('Best score: {0}\t({1})'.format(max(scores), population[np.argmax(scores)]))
        print('Av Score: {0}'.format(np.average(scores)))
        print('')
