import argparse
import ast
import os
import os.path

import numpy as np

from RL.common import logger


def str2bool(v):
    if v is True or v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v is False or v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2level(v):
    return getattr(logger, v)


def literal(v):
    return ast.literal_eval(v)


def parse():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID')
    parser.add_argument(
        '--ob_dtype', help='datatype of observations eg. uint8, float32', default='float32')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=[
                        'cnn', 'lstm', 'lnlstm', 'fc', 'ers', 'ers2', 'ers3', 'fcwithbias', 'biaslc', 'random', 'noop', 'greedy', 'ga', 'rat'], default='fc')
    parser.add_argument('--lrschedule', help='Learning rate schedule',
                        choices=['constant', 'linear'], default='constant')
    parser.add_argument('--ecschedule', help='Entropy coefficient schedule',
                        choices=['constant', 'linear'], default='constant')
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6)',
                        type=float, default=7.2)  # 5000 = 625 * 8 episodes
    # parser.add_argument('--million_frames', help='How many frames to train (/ 1e6)', type=float, default=0.71424) # 520 = 65 * 8 episodes
    parser.add_argument(
        '--num_cpu', help='Number of parallel environments', type=int, default=16)
    parser.add_argument(
        '--nsteps', help='an update happens every nsteps timesteps for each env', type=int, default=5)
    parser.add_argument(
        '--nstack', help='how many frames to stack to create one obs', type=int, default=3)
    parser.add_argument('--_lambda', help='lambda=1 => use nsteps returns. lambda=0 => use 1 step returns. intermidiate values cause averaging of various step returns. Equivalent to eligibility traces', type=float, default=0.95)
    parser.add_argument(
        '--logdir', help='logs will be saved to {logdir}/{env}/{run_no}/  . Defaults to os env variable OPENAI_LOGDIR. run_no gets incremented automatically based on existance of previous runs in {logdir}/{env}/ . No logging if logdir is not provided and the env variable is not set', default=os.getenv('OPENAI_LOGDIR'))
    parser.add_argument(
        '--saved_model', help='file from which to restore model. This file will not get overwritten when new model is saved. New models are always saved to {logdir}/{env}/{run_no}/model', default=None)
    parser.add_argument(
        '--render', help='whether or not to render the env. False by default', type=str2bool, default=False)
    parser.add_argument('--render_mode', default='human')
    parser.add_argument('--render_fps', type=float, default=None)
    parser.add_argument('--render_graphs', type=str2bool, default=True)
    parser.add_argument(
        '--no_training', help='whether to just play without training', type=str2bool, default=False)
    parser.add_argument('--mb_size', type=int, default=64)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--train_every', type=int, default=2)
    parser.add_argument('--hard_update_target', type=str2bool, default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--double_Q_learning', type=str2bool, default=False)
    parser.add_argument('--advantage_learning', type=str2bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--a_lr', type=float, default=1e-4)
    parser.add_argument('--clip_norm', type=float, default=None)
    parser.add_argument('--a_clip_norm', type=float, default=None)
    parser.add_argument('--l2_reg', type=float, default=1e-2)
    parser.add_argument('--a_l2_reg', type=float, default=0)
    parser.add_argument('--reward_scaling', type=float, default=1)
    parser.add_argument('--is_mmdp', type=str2bool, default=False)
    parser.add_argument('--softmax_actor', type=str2bool, default=False)
    parser.add_argument('--soft_constraints', type=str2bool, default=False)
    parser.add_argument('--cp_optnet', type=str2bool, default=False)
    parser.add_argument('--soft_constraints_lambda', type=float, default=10000)
    parser.add_argument('--log_norm_obs_alloc', type=str2bool, default=False)
    parser.add_argument('--log_norm_action', type=str2bool, default=False)
    parser.add_argument('--rms_norm_action', type=str2bool, default=False)
    parser.add_argument('--exploration_episodes', type=int, default=10)
    parser.add_argument('--exploration_sigma', type=float, default=None)
    parser.add_argument('--exploration_theta', type=float, default=1)
    parser.add_argument('--use_param_noise', type=str2bool, default=True)
    parser.add_argument('--use_safe_noise', type=str2bool, default=False)
    parser.add_argument('--exploit_every', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--load_every', type=int, default=100000)
    parser.add_argument('--use_ga_optimization', type=str2bool, default=False)
    parser.add_argument('--training_episodes', type=int, default=10000)
    parser.add_argument('--run_no_prefix', default='run')
    parser.add_argument('--replay_memory_gigabytes', type=float, default=None)
    parser.add_argument('--replay_buffer_length', type=int, default=1e6)
    parser.add_argument('--use_layer_norm', type=str2bool, default=True)
    parser.add_argument('--use_batch_norm', type=str2bool, default=False)
    parser.add_argument('--use_norm_actor', type=str2bool, default=True,
                        help='For DDPG: whether bn/ln, whatever is set, is to be applied to actor as well')
    parser.add_argument('--init_scale', type=float, default=3e-3)
    parser.add_argument('--nn_size', type=literal, default="[64,64]")
    parser.add_argument('--generations', type=int, default=1000)
    parser.add_argument('--population_size', type=int, default=50)
    parser.add_argument('--truncation_size', type=int, default=10)
    parser.add_argument('--mutation_sigma', type=float, default=0.02)
    parser.add_argument('--test_mode', type=str2bool, default=False)
    parser.add_argument('--analysis_mode', type=str2bool, default=False)
    parser.add_argument('--test_episodes', type=int, default=100)
    parser.add_argument('--test_seed', type=int, default=42)
    parser.add_argument('--wolpertinger_critic_train',
                        type=str2bool, default=False)
    parser.add_argument('--print_precision', type=int, default=2)
    parser.add_argument('--logger_level', type=str2level, default='INFO')
    parser.add_argument('--monitor', type=str2bool, default=False)
    parser.add_argument('--video_interval', type=int, default=None,
                        help='Capped_cubic_interval by default. Set to 0 to disable video recording. Set to N to records every N episodes')
    args = parser.parse_args()

    np.set_printoptions(precision=args.print_precision, linewidth=200)

    if args.logdir:
        logger.set_logdir(args.logdir, args.env, args.run_no_prefix)

    logger.set_level(args.logger_level)
    logger.log(str(args))
    logger.log('--------------------------------------\n')

    return args
