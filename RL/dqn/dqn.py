"""dqn.py: Implementation of Deepmind's DQN algorithm using Tensorflow and OpenAI gym.
extras:
    Double DQN
    Dueling DQN
    Observation Normalization
    DDPG style soft updated target network
    clipping, regularization etc.
    RND prediction based (episodic) intrinsic reward (single value head) - a naive implementation

original paper: https://www.nature.com/articles/nature14236

Usage:
    python3 dqn.py [env_id]
    #classic_control to know about more available environments.
    env_id: optional. default='CartPole-v0'. refer https://gym.openai.com/envs/

python dependencies:
    gym[classic_control], tensorflow, numpy

What has not been implemented yet:
    * Use of convolutional neural networks to learn atari environments from pixels.
    * Code for saving and loading the network.
"""

__author__ = "Abhinav Bhatia"
__email__ = "bhatiaabhinav93@gmail.com"
__license__ = "gpl"
__version__ = "1.0.0"

import sys

import gym  # noqa: F401
import numpy as np
import tensorflow as tf

from RL.common import logger
from RL.common.context import Context
from RL.common.utils import set_global_seeds, SimpleImageViewer, color_to_grayscale
from RL.common.experience_buffer import Experience, ExperienceBuffer
from RL.common.summaries import Summaries
from RL.dqn.rnd import RND_System
from RL.dqn.model import Brain
from PIL import Image


def dqn(env_id):
    context = Context(env_id)
    set_global_seeds(context.seed)
    env = context.make_env(env_id)  # type: gym.wrappers.Monitor
    env.seed(context.seed)
    main_brain = Brain(context, 'main_brain', False)
    target_brain = Brain(context, 'target_brain', True)
    target_brain_update_op = main_brain.tf_copy_to(
        target_brain, soft_copy=False, name='target_brain_update_op')
    target_brain_soft_update_op = main_brain.tf_copy_to(
        target_brain, soft_copy=True, name='target_brain_soft_update_op')
    viewer = SimpleImageViewer(caption=env_id + ":" + context.experiment_name)
    if context.rnd_mode:
        rnd_system = RND_System(context, 'rnd_system')

    with tf.Session() as session:
        context.session = session
        context.summaries = Summaries(session)
        session.run(tf.global_variables_initializer())
        session.run(target_brain_update_op)
        experience_buffer = ExperienceBuffer(size_in_bytes=2 * (1024 ** 3))
        for episode_id in range(context.n_episodes):
            done = False
            intrinsic_R = 0
            eval_mode = context.should_eval_episode()
            env.stats_recorder.type = 'e' if eval_mode else 't'
            obs = env.reset()
            while not done:
                frame_id = context.total_steps
                main_brain.update_running_stats([obs])
                if np.random.random() > context.epsilon or eval_mode:
                    action = main_brain.get_action([obs])[0]
                else:
                    action = context.env.action_space.sample()
                obs1, reward, done, info = env.step(action)
                if context.rnd_mode:
                    rnd_intrinsic_reward = rnd_system.get_rewards(
                        [obs1], update_stats=True)[0]
                    if frame_id > context.minimum_experience:
                        intrinsic_R += rnd_intrinsic_reward
                        reward = reward + rnd_intrinsic_reward
                experience = Experience(obs, action, reward, done, info, obs1)
                experience_buffer.add(experience)
                if context.render or episode_id % context.video_interval == 0:
                    if context.need_conv_net:
                        sensitivity = main_brain.get_Q_input_sensitivity([obs])[0]
                        sensitivity = (np.sqrt(color_to_grayscale(sensitivity)) * 255).astype(np.uint8)
                        orig = env.render(mode='rgb_array')
                        h, w = orig.shape[0], orig.shape[1]
                        sensitivity = np.asarray(Image.fromarray(sensitivity).resize((w, h), resample=Image.BILINEAR))
                        mixed = 0.8 * sensitivity + 0.2 * orig
                        viewer.imshow(mixed.astype(np.uint8))
                    else:
                        viewer.imshow(env.render(mode='rgb_array'))

                # let's train:
                if frame_id % context.train_every == 0 and frame_id > context.minimum_experience:
                    # mb_exps = list(experience_buffer.random_experiences(
                    #     context.minibatch_size))
                    # mb_states = [exp.state for exp in mb_exps]
                    # mb_actions = [exp.action for exp in mb_exps]
                    # mb_rewards = np.asarray([exp.reward for exp in mb_exps])
                    # mb_dones = np.asarray([int(exp.done) for exp in mb_exps])
                    # mb_states1 = [exp.next_state for exp in mb_exps]
                    # if not context.double_dqn:
                    #     mb_states1_V = target_brain.get_V(mb_states1)
                    # else:
                    #     mb_states1_a = main_brain.get_action(mb_states1)
                    #     mb_states1_Q = target_brain.get_Q(mb_states1)
                    #     mb_states1_V = np.asarray(
                    #         [mb_states1_Q[exp_id, mb_states1_a[exp_id]] for exp_id in range(context.minibatch_size)])
                    # mb_gammas = (1 - mb_dones) * context.gamma
                    states, actions, rewards, dones, infos, states1 = experience_buffer.random_rollouts_unzipped(int(context.minibatch_size / context.nsteps), context.nsteps)
                    for step in range(context.nsteps - 1, -1, -1):
                        next_r = target_brain.get_V(states1[:, -1]) if step == context.nsteps - 1 else rewards[:, step + 1]
                        rewards[:, step] = rewards[:, step] + (1 - dones[:, step]) * context.gamma * next_r
                    mb_states = np.reshape(states, [context.minibatch_size] + list(env.observation_space.shape))
                    mb_rewards = rewards.flatten()
                    mb_actions = actions.flatten()
                    mb_desired_Q = main_brain.get_Q(mb_states)
                    all_rows = np.arange(context.minibatch_size)
                    mb_error = mb_rewards - mb_desired_Q[all_rows, mb_actions]
                    if context.clip_td_error:
                        mb_error = np.clip(mb_error, -1, 1)
                    mb_desired_Q[all_rows, mb_actions] = mb_desired_Q[all_rows, mb_actions] + mb_error
                    # for exp_id in range(context.minibatch_size):
                    #     error = mb_rewards[exp_id] - mb_desired_Q[exp_id, mb_actions[exp_id]]
                    #     if context.clip_td_error:
                    #         error = np.clip(error, -1, 1)
                    #     mb_desired_Q[exp_id, mb_actions[exp_id]] = mb_desired_Q[exp_id, mb_actions[exp_id]] + error
                    _, mse = main_brain.train(mb_states, mb_desired_Q)
                    # half_mb_size = int(context.minibatch_size / 2)
                    # mb_exps_random = list(experience_buffer.random_experiences(half_mb_size))
                    # mb_next_states_random = [exp.next_state for exp in mb_exps_random]
                    # mb_desired_tc = np.zeros([context.minibatch_size, 2])
                    # mb_desired_tc[0:half_mb_size, 0] = 1
                    # mb_desired_tc[half_mb_size:context.minibatch_size, 1] = 1
                    # mb_next_states = mb_next_states_random + mb_states1[0:half_mb_size]
                    # assert len(mb_states) == len(mb_next_states)
                    # main_brain.train_transitions(mb_states, mb_next_states, mb_desired_tc)
                    # if context.rnd_mode:
                    #     _, rnd_mse = rnd_system.train(mb_states1)

                # update target network:
                if not context.ddpg_target_network_update_mode:
                    if frame_id % context.target_network_update_interval == 0:
                        session.run(target_brain_update_op)
                        logger.log('Target network updated')
                else:
                    session.run(target_brain_soft_update_op)

                obs = obs1
            context.intrinsic_returns.append(intrinsic_R)
            context.log_stats(
                average_over=100, end='\n' if episode_id % 50 == 0 else '\r')
            if episode_id % 50 == 0:
                main_brain.save()
                main_brain.save(suffix=str(episode_id))
                logger.log("model saved")

        logger.log('---------------Done--------------------')
        context.log_stats(average_over=context.n_episodes)


if __name__ == '__main__':
    env_id = sys.argv[1] if len(sys.argv) > 1 else Context.default_env_id
    dqn(env_id)
