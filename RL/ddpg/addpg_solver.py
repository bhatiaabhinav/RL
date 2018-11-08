"""
dualing network style advantageous DDPG
"""
import os
import os.path
import random
import time
from typing import List  # noqa: F401

import gym
import numpy as np
import tensorflow as tf

import gym_BSS  # noqa: F401
import gym_ERSLE  # noqa: F401
from RL.common import logger
from RL.common.atari_wrappers import (BreakoutContinuousActionWrapper,
                                      ClipRewardEnv, EpisodicLifeEnv,
                                      FireResetEnv, FrameStack, MaxEnv,
                                      NoopResetEnv, SkipAndFrameStack, SkipEnv,
                                      WarpFrame)
from RL.common.experience_buffer import Experience, ExperienceBuffer
from RL.common.noise import NormalNoise  # noqa: F401
from RL.common.noise import OrnsteinUhlenbeckActionNoise
from RL.common.utils import mutated_ers, mutated_gaussian, normalize
from RL.common.wrappers import (ActionSpaceNormalizeWrapper, CartPoleWrapper,
                                LinearFrameStackWrapper)
from RL.ddpg.addpg_ac_model import DDPG_Model
from RL.reco_rl.wrappers import (BSStoMMDPWrapper, ERStoMMDPWrapper,
                                 MMDPActionSpaceNormalizerWrapper,
                                 MMDPActionWrapper, MMDPObsNormalizeWrapper,
                                 MMDPObsStackWrapper)
from RL.common.plot_renderer import PlotRenderer


def get_action(model: DDPG_Model, obs, env: gym.Env, action_noise, use_param_noise, exploit_mode, normalize_action, log, f):
    if not use_param_noise or exploit_mode:
        # get exploitative action
        if log:
            if model.main.advantage_learning:
                a, V, A, Q = model.main.get_a_V_A_Q([obs])
                a, V, A, Q = a[0], V[0], A[0], Q[0]
                model.summaries.write_summaries(
                    {'frame_A': A, 'frame_Q': Q}, f)
            else:
                a, Q = model.main.get_a_Q([obs])
                a, Q = a[0], Q[0]
                model.summaries.write_summaries({'frame_Q': Q}, f)
        else:
            a = model.main.get_a([obs])[0]
    if not exploit_mode:
        if use_param_noise:
            a = model.noisy.get_a([obs])[0]
        else:
            a += action_noise()
            if normalize_action:
                a = normalize(a)
            else:
                a = np.clip(a, -1, 1)
                a = env.action_space.low + \
                    (env.action_space.high - env.action_space.low) * (a + 1) / 2
    return a


def train(model: DDPG_Model, experience_buffer: ExperienceBuffer, global_frame_index, mb_size, gamma, double_Q_learning=False, advantage_learning=False, hard_update_target=False, train_every=1, tau=0.001, log=False):
    mb = list(experience_buffer.random_experiences(
        count=mb_size))  # type: List[Experience]
    s, a, s_next, r, d = [e.state for e in mb], [e.action for e in mb], [
        e.next_state for e in mb], np.asarray([e.reward for e in mb]), np.asarray([int(e.done) for e in mb])
    ɣ = (1 - d) * gamma

    if not double_Q_learning:
        # Single Q learning:
        if advantage_learning:
            # A(s,a) <- r + ɣ * max[_Q(s_next, _)] - max[_Q(s, _)]
            # V(s) <- max[_Q(s, _)]
            old_max_Q_s = model.target.max_Q(s)
            adv_s_a = r + ɣ * model.target.max_Q(s_next) - old_max_Q_s
            v_s = old_max_Q_s
        else:
            # Q(s,a) <- r + ɣ * max[_Q(s_next, _)]
            q_s_a = r + ɣ * model.target.max_Q(s_next)
    else:
        if advantage_learning:
            # Double Q learning:
            # A(s,a) <- r + ɣ * _Q(s_next, argmax[Q(s_next, _)]) - _Q(s, argmax(s, _))
            # V(s) <- _Q(s, argmax[Q(s, _)])
            old_Q_s1_argmax_Q_s1 = model.target.Q(
                s_next, model.main.argmax_Q(s_next))
            old_Q_s_argmax_Q_s = model.target.Q(model.main.argmax_Q(s))
            adv_s_a = r + ɣ * old_Q_s1_argmax_Q_s1 - old_Q_s_argmax_Q_s
            v_s = old_Q_s_argmax_Q_s
        else:
            # Q <- r + ɣ * _Q(s_next, argmax[Q(s_next, _)])
            q_s_a = r + ɣ * \
                model.target.Q(s_next, model.main.argmax_Q(s_next))

    if advantage_learning:
        _, A_mse = model.main.train_A(states=s, actions=a, target_A=adv_s_a)
        _, V_mse = model.main.train_V(s, v_s)
        _, av_max_A, av_infeasibility = model.main.train_a(s)
    else:
        _, Q_mse = model.main.train_Q(states=s, actions=a, target_Q=q_s_a)
        _, av_max_Q, av_infeasibility = model.main.train_a(s)

    f = global_frame_index
    if hard_update_target:
        if f % int(train_every / tau) == 0:  # every train_every/tau steps
            model.target.update_from_main_network()
    else:
        model.target.soft_update_from_main_network()

    if log:
        if advantage_learning:
            model.summaries.write_summaries({
                'A_mse': A_mse,
                'V_mse': V_mse,
                'av_max_A': av_max_A,
                'av_infeasibility': av_infeasibility
            }, f)
        else:
            model.summaries.write_summaries({
                'Q_mse': Q_mse,
                'av_max_Q': av_max_Q,
                'av_infeasibility': av_infeasibility
            }, f)


def ga_optimize_action(model: DDPG_Model, s, a_start, generations, population_size, truncation_size, mutation_fn, mutation_rate):
    cur_gen, prev_gen = [None] * population_size, [None] * population_size
    for g in range(generations):
        for idx in range(population_size):
            if g == 0:
                if idx == 0:
                    cur_gen[idx] = a_start
                else:
                    cur_gen[idx] = mutation_fn(
                        a_start, mutation_rate=mutation_rate)
            else:
                if idx == 0:
                    # for elitism. i.e. leave the top parent unmodified
                    cur_gen[idx] = prev_gen[idx]
                else:
                    # select parent from idx [0, T)
                    parent_idx = np.random.randint(0, truncation_size)
                    cur_gen[idx] = mutation_fn(
                        prev_gen[parent_idx], mutation_rate=mutation_rate)
        # eval the generation:
        fitness = model.main.Q([s] * population_size, cur_gen)
        # sort the population in decreasing order by fitness
        cur_gen = [a for a, q in sorted(
            zip(cur_gen, fitness), key=lambda pair:pair[1], reverse=True)]
        prev_gen = cur_gen
        cur_gen = [None] * population_size
    return prev_gen[0]


def ga_optimize_actor(model: DDPG_Model, states, mutation_fn, mutation_rate, train_steps=1000):
    s = states
    a = model.main.argmax_Q(s)
    discovered_best_a = []
    new_discoveries_count = 0
    batch_size = len(states)
    for idx in range(batch_size):
        _s = s[idx]
        _a = a[idx]
        _discovered_best_a = ga_optimize_action(
            model, _s, _a, generations=4, population_size=64, truncation_size=16, mutation_fn=mutation_fn, mutation_rate=mutation_rate)
        discovered_best_a.append(_discovered_best_a)
        if not np.array_equiv(_a, _discovered_best_a):
            new_discoveries_count += 1
    logger.logkv('new a discoveries', new_discoveries_count)
    for step in range(train_steps):
        model.main.train_a_supervised(s, discovered_best_a)


def reset_noise(model: DDPG_Model, noise, use_param_noise, use_safe_noise, experience_buffer: ExperienceBuffer, mb_size, episode_no):
    if use_param_noise:
        if len(experience_buffer) >= mb_size:
            mb_states = experience_buffer.random_states(mb_size)
            if use_safe_noise:
                model.noisy.noisy_update_from_main_network(
                    model.noisy.generate_safe_noise(mb_states))
            else:
                model.noisy.noisy_update_from_main_network(
                    model.noisy.generate_normal_param_noise())
            divergence = model.noisy.get_divergence(mb_states)
            logger.logkv('Exploration Divergence', divergence)
            model.summaries.write_summaries(
                {'divergence': divergence}, episode_no)
            model.noisy.adapt_sigma(divergence)
        else:
            model.noisy.noisy_update_from_main_network(
                model.noisy.generate_normal_param_noise())
        model.summaries.write_summaries(
            {'adaptive_sigma': model.noisy.adaptive_sigma}, episode_no)
    else:
        noise.reset()


def load_model(model: DDPG_Model, load_path):
    try:
        model.main.load(load_path)
        logger.log('model loaded')
    except Exception as ex:
        logger.log('Failed to load model. Reason = {0}'.format(
            ex), level=logger.ERROR)


def ddpg(sys_args_dict, sess, env_id, wrappers, learning=False, actor=None, seed=0, learning_env_seed=0,
         test_env_seed=42, learning_episodes=40000, test_episodes=100, exploration_episodes=10, train_every=1,
         mb_size=64, use_safe_noise=False, replay_buffer_length=1e6, replay_memory_size_in_bytes=None, use_param_noise=False,
         init_scale=1e-3, reward_scaling=1, Noise_type=OrnsteinUhlenbeckActionNoise, exploration_sigma=0.2, exploration_theta=1, exploit_every=10,
         gamma=0.99, double_Q_learning=False, advantage_learning=False, hard_update_target=False, tau=0.001, use_ga_optimization=False, render=False, render_graphs=False,
         render_mode='human', render_fps=60, log_every=100, save_every=50, load_every=1000000, save_path=None, load_path=None, is_mmdp=False,
         softmax_actor=False, wolpertinger_critic_train=False, monitor=True, video_interval=50, **kwargs):
    set_global_seeds(seed)
    env = gym.make(env_id)  # type: gym.Env
    for W in wrappers:
        env = W(env)  # type: gym.Wrapper
    if monitor:
        from RL.common.utils import my_video_schedule
        env = gym.wrappers.Monitor(env, os.path.join(logger.get_dir(), 'monitor'), video_callable=lambda ep_id: my_video_schedule(
            ep_id, learning_episodes if learning else test_episodes, video_interval))
    if actor is None:
        sys_args_dict["ob_space"] = env.observation_space
        sys_args_dict["ac_space"] = env.action_space
        sys_args_dict["constraints"] = env.metadata.get('constraints', None)
        # sys_args_dict["log_transform_max_x"] = env.metadata.get(
        #     'max_alloc', None)
        # sys_args_dict["log_transform_t"] = env.metadata.get(
        #     'alloc_log_transform_t', None)
        if exploration_sigma is None:
            exploration_sigma = 1 / env.metadata.get('nresources', 5)
            logger.log('exploration_sigma has been set as {0}'.format(
                exploration_sigma))
        sys_args_dict['target_divergence'] = exploration_sigma
        model = DDPG_Model(sess, use_param_noise, sys_args_dict)
        sess.run(tf.global_variables_initializer())
        model.summaries.setup_scalar_summaries(['V_mse', 'A_mse', 'Q_mse', 'av_max_A', 'av_max_Q', 'av_infeasibility',
                                                'frame_Q', 'frame_A', 'R', 'ep_length', 'R_exploit', 'blip_R_exploit',
                                                'divergence', 'adaptive_sigma', 'wrap_effect'])
        model.summaries.setup_histogram_summaries(
            ['ep_av_action'])
    else:
        model = actor
    if load_path:
        load_model(model, load_path)
    model.target.update_from_main_network()
    noise = None
    experience_buffer = None
    if is_mmdp:
        if wolpertinger_critic_train:
            logger.log("Wolpertinger mode")
        action_wrapper = MMDPActionWrapper(
            env, assume_feasible_action_input=softmax_actor)
        logger.log(
            'Constrained Softmax Layer Mode' if softmax_actor else 'Soft Constraints Mode')
    if learning:
        experience_buffer = ExperienceBuffer(
            length=replay_buffer_length, size_in_bytes=replay_memory_size_in_bytes)
        noise = None if use_param_noise else Noise_type(mu=np.zeros(
            env.action_space.shape), sigma=exploration_sigma, theta=exploration_theta)

    logdir = os.path.basename(os.path.normpath(logger.get_dir()))
    window_name = env_id + ":" + logdir
    action_renderer = PlotRenderer(600, 600, 'Episode average action', xlabel='base_id',
                                   ylabel='Action', window_caption=window_name, concat_title_with_caption=False)
    action_renderer.plot(list(range(env.action_space.shape[0])), [
                         0] * env.action_space.shape[0])
    score_renderer = PlotRenderer(600, 600, 'Average Reward per Episode', xlabel='Episode',
                                  ylabel='Reward', window_caption=window_name, concat_title_with_caption=False, smoothing=100)
    score_renderer.plot([], [], 'b-', [], [], 'g--')
    Rs, exploit_Rs, exploit_blip_Rs, f = [], [], [], 0
    env.seed(learning_env_seed if learning else test_env_seed)
    for ep in range(learning_episodes if learning else test_episodes):
        obs, d, R, blip_R, ep_l, ep_sum_a = env.reset(), False, 0, 0, 0, 0
        exploit_mode = (ep % exploit_every == 0) or not learning
        if not exploit_mode:
            reset_noise(model, noise, use_param_noise, use_safe_noise,
                        experience_buffer, mb_size, ep)
        while not d:
            should_log = (f % log_every == 0)
            if learning and ep >= exploration_episodes and f % train_every == 0:
                train(model, experience_buffer, f, mb_size, gamma, double_Q_learning=double_Q_learning,
                      advantage_learning=advantage_learning, hard_update_target=hard_update_target, train_every=train_every, tau=tau, log=should_log)
                if use_ga_optimization and f % (50 * train_every) == 0:
                    mutation_fn = mutated_ers if (
                        'ERS' in env_id or 'BSS' in env_id) else mutated_gaussian
                    ga_optimize_actor(model, experience_buffer.random_states(
                        mb_size), mutation_fn, exploration_sigma, train_steps=100)
            raw_action = get_action(model=model, obs=obs, env=env, action_noise=noise,
                                    use_param_noise=use_param_noise, exploit_mode=exploit_mode, normalize_action=softmax_actor, log=should_log, f=f)
            if is_mmdp:
                action, wrap_effect = action_wrapper.wrap(raw_action)
                if should_log:
                    model.summaries.write_summaries(
                        {'wrap_effect': wrap_effect}, f)
            else:
                action = raw_action
            ep_sum_a = ep_sum_a + action
            obs_, r, d, _ = env.step(action)
            r = r * reward_scaling
            if render:
                env.render(mode=render_mode)
                if render_fps is not None:
                    time.sleep(1 / render_fps)
            if learning:
                if wolpertinger_critic_train:
                    critic_training_action = action
                else:
                    critic_training_action = raw_action
                experience_buffer.add(Experience(
                    obs, critic_training_action, r, d, _, obs_))
                model.main.update_running_ob_stats(obs)
                model.main.update_running_ac_stats(critic_training_action)
            obs, R, f, ep_l = obs_, R + r, f + 1, ep_l + 1
            if 'blip_reward' in _:
                blip_R += _['blip_reward']
        R = R / reward_scaling
        Rs.append(R)
        if exploit_mode:
            exploit_Rs.append(R)
            exploit_blip_Rs.append(blip_R)
        logger.logkvs({'Episode': ep, 'Reward': R, 'Exploited': exploit_mode, 'Blip_Reward': blip_R, 'Length': ep_l, 'Average Reward': np.average(
            Rs[-100:]), 'Exploit Average Reward': np.average(exploit_Rs[-100:]), 'Exploit Average Blip Reward': np.average(exploit_blip_Rs[-100:])})
        logger.dump_tabular()
        ep_av_a = ep_sum_a * env.metadata.get('nresources', 1) / ep_l
        logger.log('Average action: {0}'.format(ep_av_a))
        if render_graphs:
            score_renderer.append_and_render([exploit_Rs[-1], Rs[-1]])
            action_renderer.update_and_render([[list(range(25)), ep_av_a]])
        model.summaries.write_summaries(
            {'R': R, 'R_exploit': exploit_Rs[-1], 'blip_R_exploit': exploit_blip_Rs[-1], 'ep_length': ep_l, 'ep_av_action': ep_av_a}, ep)
        if save_path and ep % save_every == 0:
            model.main.save(save_path)
            logger.log('model saved')
        if not learning and load_path is not None and ep % load_every == 0:
            load_model(model, load_path)

    env.close()
    logger.log('Average reward per episode: {0}'.format(np.average(Rs)))
    logger.log('Std reward per episode: {0}'.format(np.std(Rs)))
    logger.log('Exploitation average reward per episode: {0}'.format(
        np.average(exploit_Rs)))
    logger.log('Exploitation std reward per episode: {0}'.format(
        np.std(exploit_Rs)))
    logger.log('Exploitation average blip reward per episode: {0}'.format(
        np.average(exploit_blip_Rs)))
    logger.log('Exploitation std blip reward per episode: {0}'.format(
        np.std(exploit_blip_Rs)))
    return model


def main(sys_args_dict, seed=0, learning_env_seed=0, test_env_seed=42, test_mode=False, analysis_mode=False, save_path=None, load_path=None, **kwargs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if not (test_mode or analysis_mode):
            logger.log('Training actor. seed={0}. learning_env_seed={1}'.format(
                seed, learning_env_seed))
            model = ddpg(
                sys_args_dict, sess, learning=True, actor=None, **sys_args_dict)
            model.main.save(save_path)
            logger.log(
                'Testing actor. test_env_seed={0}'.format(test_env_seed))
            ddpg(sys_args_dict, sess, learning=False, actor=model,
                 **dict(sys_args_dict, save_path=None, load_path=None))
            logger.log('Testing done. Seeds were seed={0}. learning_env_seed={1}. test_env_seed={2}'.format(
                seed, learning_env_seed, test_env_seed))
            # logger.log('Analysing model')
            # analyse_q(sys_args_dict, sess, actor=actor, **
            #           dict(sys_args_dict, load_path=None))
            # logger.log("Analysis done. Results saved to logdir.")
        if test_mode:
            logger.log(
                'Testing actor. test_env_seed={0}'.format(test_env_seed))
            assert load_path is not None, "Please provide a saved model"
            ddpg(sys_args_dict, sess, learning=False, **
                 dict(sys_args_dict, save_path=None))
            logger.log('Testing done. Seeds were seed={0}. learning_env_seed={1}. test_env_seed={2}'.format(
                seed, learning_env_seed, test_env_seed))
        if analysis_mode:
            logger.log('Analysing model')
            assert load_path is not None, "Please provide a saved model"
            # analyse_q(sys_args_dict, sess, actor=None, **sys_args_dict)
            raise NotImplementedError()
            logger.log("Analysis done. Results saved to logdir.")

        logger.log('-------------------------------------------------\n')


def set_global_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == '__main__':
    # config = tf.ConfigProto(device_count={'GPU': 0})
    from RL.common.args import parse
    args = parse()
    logger.log('env_id: ' + args.env)
    logger.log('Seed: {0}'.format(args.seed))
    set_global_seeds(args.seed)
    kwargs = vars(args).copy()
    kwargs['env_id'] = args.env
    kwargs['wrappers'] = []
    if args.replay_memory_gigabytes is not None:
        kwargs['replay_memory_size_in_bytes'] = args.replay_memory_gigabytes * 2**30
    else:
        kwargs['replay_memory_size_in_bytes'] = None
    kwargs['Noise_type'] = OrnsteinUhlenbeckActionNoise
    kwargs['learning_env_seed'] = args.seed
    kwargs['learning_episodes'] = args.training_episodes
    kwargs['test_env_seed'] = args.test_seed
    assert not (
        kwargs['use_layer_norm'] and kwargs['use_batch_norm']), "Cannot use both layer norm and batch norm"
    kwargs['save_path'] = os.path.join(logger.get_dir(), "model")
    kwargs['load_path'] = args.saved_model
    MMDPObsStackWrapper.k = args.nstack
    FrameStack.k = args.nstack
    LinearFrameStackWrapper.k = args.nstack
    if kwargs['soft_constraints']:
        # assert not kwargs['wolpertinger_critic_train'], "Wolpertinger cannot be used with soft constraints mode"
        assert not kwargs['softmax_actor'], "Cannot have both hard constraints and soft constraints"
    if 'ERSEnv-ca' in kwargs['env_id']:
        kwargs['wrappers'] = [ERStoMMDPWrapper, MMDPActionSpaceNormalizerWrapper,
                              MMDPObsNormalizeWrapper, MMDPObsStackWrapper]
        kwargs['is_mmdp'] = True
    elif 'BSSEnv' in kwargs['env_id']:
        kwargs['wrappers'] = [BSStoMMDPWrapper, MMDPActionSpaceNormalizerWrapper,
                              MMDPObsNormalizeWrapper, MMDPObsStackWrapper]
        kwargs['is_mmdp'] = True
    elif 'Pole' in kwargs['env_id']:
        kwargs['wrappers'] = [CartPoleWrapper]
    elif 'NoFrameskip' in kwargs['env_id']:
        kwargs['wrappers'] = [EpisodicLifeEnv, NoopResetEnv, MaxEnv, FireResetEnv,
                              WarpFrame, SkipAndFrameStack, ClipRewardEnv, BreakoutContinuousActionWrapper]
    elif 'CarRacing' in kwargs['env_id']:
        kwargs['wrappers'] = [SkipEnv, WarpFrame,
                              FrameStack, ActionSpaceNormalizeWrapper]
    elif 'Bipedal' in kwargs['env_id']:
        from baselines.ers.wrappers import BipedalWrapper
        kwargs['wrappers'] = [ActionSpaceNormalizeWrapper, BipedalWrapper]
    else:
        kwargs['wrappers'] = [LinearFrameStackWrapper,
                              ActionSpaceNormalizeWrapper]
    main(kwargs, **kwargs)
