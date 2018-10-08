import math
import os
import os.path

import gym
import joblib
import numpy as np
import tensorflow as tf

import gym_ERSLE  # noqa: F401
from RL.common import logger
from RL.common.atari_wrappers import FrameStack
from RL.common.policy_tester import test
from RL.common.utils import conv, conv_to_fc, fc, set_global_seeds
from RL.common.vec_env.subproc_vec_env import SubprocVecEnv


class Actor:
    instance = None

    def __init__(self, session: tf.Session, name, ob_shape, ac_shape,
                 ob_dtype='float32', softmax_output=True,
                 nn_size=[400, 300, 300], init_scale=1e-4):
        assert len(ac_shape) == 1
        self.session = session
        self.name = name
        self.ac_shape = ac_shape
        with tf.variable_scope(name):
            self.states_feed = tf.placeholder(
                dtype=ob_dtype, shape=[None] + list(ob_shape))
            states_feed = self.states_feed
            with tf.variable_scope('a'):
                # conv layers go here
                if len(ob_shape) > 1:
                    inp = tf.cast(states_feed, tf.float32) / 255.
                    if ob_shape[0] >= 60:
                        a_c1 = conv(inp, 'a_c1', nf=32, rf=8,
                                    stride=4, init_scale=np.sqrt(2))
                    a_c2 = conv(a_c1 if ob_shape[0] >= 60 else inp, 'a_c2',
                                nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
                    a_c3 = conv(a_c2, 'a_c3', nf=64, rf=3,
                                stride=1, init_scale=np.sqrt(2))
                    states_flat = conv_to_fc(a_c3)
                else:
                    states_flat = states_feed
                a_h1 = fc(states_flat, 'a_h1', nh=nn_size[0])
                a_h2 = fc(a_h1, 'a_h2', nh=nn_size[1])
                if softmax_output:
                    a = fc(a_h2, 'a', nh=ac_shape[0],
                           act=lambda x: x, init_scale=init_scale)
                    exp = tf.exp(a - tf.reduce_max(a, axis=-1, keep_dims=True))
                    a = exp / tf.reduce_sum(exp, axis=-1, keep_dims=True)
                else:
                    a = fc(
                        a_h2, 'a', nh=ac_shape[0], act=tf.nn.tanh,
                        init_scale=init_scale)
                self.a = a

        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{0}'.format(name))
        self.load_placeholders = []
        self.load_ops = []
        for p in self.params:
            p_placeholder = tf.placeholder(
                shape=p.shape.as_list(), dtype=tf.float32)
            self.load_placeholders.append(p_placeholder)
            self.load_ops.append(p.assign(p_placeholder))

        self.writer = tf.summary.FileWriter(
            'summary/ga_policy_solver', self.session.graph)

    def get_instance():
        return Actor.instance

    def apply_params(self, params):
        feed_dict = {}
        for p, p_placeholder in zip(params, self.load_placeholders):
            feed_dict[p_placeholder] = p
        self.session.run(self.load_ops, feed_dict=feed_dict)

    def get_params(self):
        return self.session.run(self.params)

    def act(self, states):
        return self.session.run(self.a, feed_dict={self.states_feed: states})


def random_params():
    params = []
    for matrix in Actor.get_instance().get_params():
        random_matrix = np.random.standard_normal(size=np.shape(matrix))
        params.append(random_matrix)
    return params


def mutated(params, mutation_sigma, interpret_as_max_sigma=True):
    return_params = []
    sigma = mutation_sigma * np.random.rand() if interpret_as_max_sigma\
        else mutation_sigma
    for p in params:
        mutated_p = p + sigma * np.random.standard_normal(size=np.shape(p))
        return_params.append(mutated_p)
    return return_params


def reseed(envs, seed):
    seeds = [seed + i for i in range(envs.num_envs)]
    print('(Re)seeding the envs with seeds {0}'.format(seeds))
    envs.seed(seeds)


def eval(params, envs, seed):
    Actor.get_instance().apply_params(params)
    av_r = 0
    reseed(envs, seed)
    obs = envs.reset()
    done = False
    while not done:
        actions = Actor.get_instance().act(obs)
        # print(np.round(24 * actions, decimals=2))
        obs, r, dones, infos = envs.step(actions)
        av_r += np.average(r)
        done = any(dones)
    print('Model_score: {0}'.format(av_r))
    return av_r


def eval_generation(generation, envs, seed):
    fitness = [eval(params, envs, seed) for params in generation]
    return fitness


def save_params(params, save_path):
    from baselines.a2c.utils import make_path
    make_path(os.path.dirname(save_path))
    joblib.dump(params, save_path)


def save_generation(generation, save_folder):
    for i in range(len(generation)):
        path = os.path.join(save_folder, 'model_{0}'.format(i))
        save_params(generation[i], path)


def load_params(load_path):
    return joblib.load(load_path)


def load_generation(generation, load_folder):
    for i in range(len(generation)):
        path = os.path.join(load_folder, 'model_{0}'.format(i))
        generation[i] = load_params(path)


def optimize(envs, seed, generations, population_size, truncation_size,
             mutation_sigma, saved_model=None):
    cur_gen, prev_gen = [None] * population_size, [None] * population_size
    top_scores, average_scores, min_scores = [], [], []
    for g in range(generations):
        for idx in range(population_size):
            if g == 0:
                cur_gen[idx] = mutated(Actor.get_instance().get_params(),
                                       mutation_sigma,
                                       interpret_as_max_sigma=False)
            else:
                if idx == 0:
                    # for elitism. i.e. leave the top parent unmodified
                    cur_gen[idx] = prev_gen[idx]
                else:
                    # select parent from idx [0, T)
                    parent_idx = np.random.randint(0, truncation_size)
                    print('selected parent {0}'.format(parent_idx))
                    cur_gen[idx] = mutated(
                        prev_gen[parent_idx], mutation_sigma)
        if g == 0 and saved_model:
            load_generation(cur_gen, saved_model)
            print('Loaded saved generation')
        print('Evaluating current generation')
        fitness = eval_generation(cur_gen, envs, seed + g * envs.num_envs)
        top_scores.append(max(fitness))
        average_scores.append(np.average(fitness))
        min_scores.append(min(fitness))
        last_n = math.ceil(100 / envs.num_envs)
        moving_av_window_size = last_n * envs.num_envs
        print('Generation: {0}\tAv_Top_score: {1}\tAv_Av_score: {2}\t\
              Av_Min_score: {3}\t(Av_window_size: {4})'.format(
            g, np.average(top_scores[-last_n:]),
            np.average(average_scores[-last_n:]),
            np.average(min_scores[-last_n:]), moving_av_window_size),
            flush=True)
        # sort the population in decreasing order by fitness
        cur_gen = [param for param, score in sorted(
            zip(cur_gen, fitness), key=lambda pair:pair[1], reverse=True)]
        # save the population
        if logger.get_dir():
            save_generation(cur_gen, os.path.join(logger.get_dir(), 'model'))
            print('Saved generation')
        prev_gen = cur_gen
        cur_gen = [None] * population_size
    return prev_gen[0]


def make_ers_env(id, wrappers=[]):
    env = gym.make(id)
    if 'im' in id:
        wrappers = [FrameStack] + wrappers
    env = gym.make(id)
    for W in wrappers:
        env = W(env)
    return env


def main():
    from baselines.ers.args import parse
    args = parse()

    # ---------- init ------------------------------------
    set_global_seeds(args.seed)
    if 'im' in args.env:
        args.ob_dtype = 'uint8'
    env = make_ers_env(args.env)

    # --------- create computation graph ------------------
    session = tf.Session()
    best_params = None
    Actor.instance = Actor(
        session, "actor", env.observation_space.shape,
        env.action_space.shape, args.ob_dtype)
    session.run(tf.global_variables_initializer())

    # ---------- train -------------------------------------
    if not args.test_mode:
        def make_env(rank):
            def _thunk():
                env = make_ers_env(args.env)
                env.seed(args.seed + rank)
                return env
            return _thunk
        envs = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])
        best_params = optimize(envs, args.seed, args.generations,
                               args.population_size,
                               args.truncation_size, args.mutation_sigma,
                               saved_model=args.saved_model)
        envs.close()

    # ----------- now test ---------------------------------
    if best_params is None and args.saved_model:
        best_params = load_params(args.saved_model)
    if not best_params:
        print('Please either turn off test_mode or provide a saved model')
    else:
        actor = Actor.get_instance()  # type: Actor
        actor.apply_params(best_params)
        test(env=env, actor=actor, seed=args.test_seed,
             episodes=args.test_episodes, render=args.render)

    # ------------ clean up --------------------------------
    env.close()
    session.close()


if __name__ == '__main__':
    main()
