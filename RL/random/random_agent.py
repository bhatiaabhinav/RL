import os

import gym
import gym.wrappers
import numpy as np

from baselines import logger
from baselines.ers.args import parse
from baselines.ers.utils import my_video_schedule

args = parse()
env = gym.make(args.env)  # type: gym.Env

if args.monitor:
    env = gym.wrappers.Monitor(env, os.path.join(
        logger.get_dir(), 'monitor'), force=True, video_callable=lambda ep_id: my_video_schedule(ep_id, args.test_episodes, args.video_interval))

env.seed(args.test_seed)
for episode in range(args.test_episodes):
    R, L, done = 0, 0, False
    obs = env.reset()
    while not done:
        obs, r, done, info = env.step(env.action_space.sample())
        R += r
        L += 1
    logger.logkvs({'Episode': episode, 'Reward': R, 'Length': L})
    logger.dump_tabular()

if args.monitor:
    logger.log('-----stats------')
    logger.logkvs({
        'Episodes': args.test_episodes,
        'Timesteps': env.stats_recorder.total_steps,
        'Average reward per episode': np.mean(env.stats_recorder.episode_rewards),
        'Average length per episode': np.mean(env.stats_recorder.episode_lengths)
    })
    logger.dumpkvs()

logger.log('Done')
