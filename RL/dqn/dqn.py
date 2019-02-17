"""dqn.py: Implementation of Deepmind's DQN algorithm using Tensorflow and OpenAI gym.
extras:
    Double DQN
    Dueling DQN
    Observation Normalization
    DDPG style soft updated target network
    clipping, regularization etc.
    random n-step rollouts are sampled from experience replay instead of point experiences
    RND prediction based (episodic) intrinsic reward (single value head) - a naive implementation

original paper: https://www.nature.com/articles/nature14236

Usage:
    python3 dqn.py --env_id=[env_id]
    # env_id can be any environment with discrete action space.
    env_id: optional. default='CartPole-v0'. refer https://gym.openai.com/envs/

python dependencies:
    gym[classic_control], tensorflow, numpy
"""

__author__ = "Abhinav Bhatia"
__email__ = "bhatiaabhinav93@gmail.com"
__license__ = "gpl"
__version__ = "1.0.0"

import gym  # noqa: F401
import numpy as np
import tensorflow as tf
from PIL import Image

from RL.common import logger
from RL.common.atari_wrappers import wrap_atari
from RL.common.context import (Agent, Context, PygletLoop, RLRunner,
                               SeedingAgent)
from RL.common.experience_buffer import Experience, ExperienceBuffer
from RL.common.summaries import Summaries
from RL.common.utils import (ImagePygletWingow, SimpleImageViewer,
                             color_to_grayscale, set_global_seeds)
from RL.common.wrappers import MaxEpisodeStepsWrapper
from RL.dqn.model import Brain
from RL.dqn.rnd import RND_System


class DQNAgent(Agent):
    def __init__(self, context: Context, name):
        super().__init__(context, name)
        self.main_brain = Brain(context, '{0}/main_brain'.format(name), False)
        self.target_brain = Brain(
            context, '{0}/target_brain'.format(name), True)
        self.target_brain_update_op = self.main_brain.tf_copy_to(
            self.target_brain, soft_copy=False, name='{0}/target_brain_update_op'.format(name))
        self.target_brain_soft_update_op = self.main_brain.tf_copy_to(
            self.target_brain, soft_copy=True, name='{0}/target_brain_soft_update_op'.format(name))
        if context.experience_buffer_megabytes is None:
            self.experience_buffer = ExperienceBuffer(
                length=context.experience_buffer_length)
        else:
            self.experience_buffer = ExperienceBuffer(
                size_in_bytes=context.experience_buffer_megabytes * (1024**2))

    def start(self):
        if self.context.load_model_dir is not None:
            self.main_brain.load(
                filename=self.name.replace('/', '_') + '_model')
        self.context.session.run(self.target_brain_update_op)

    def pre_act(self):
        self.main_brain.update_running_stats([self.context.frame_obs])

    def act(self):
        r = np.random.random()
        if r > self.context.epsilon:
            action = self.main_brain.get_action([self.context.frame_obs])[0]
        else:
            action = self.context.env.action_space.sample()
        return action

    def post_act(self):
        context = self.context
        experience = Experience(context.frame_obs, context.frame_action, context.frame_reward,
                                context.frame_done, context.frame_info, context.frame_obs_next)
        self.experience_buffer.add(experience)
        if context.frame_id % context.train_every == 0 and context.frame_id > context.minimum_experience:
            states, actions, rewards, dones, infos, states1 = self.experience_buffer.random_rollouts_unzipped(
                int(context.minibatch_size / context.nsteps), context.nsteps)
            for step in range(context.nsteps - 1, -1, -1):
                next_r = self.target_brain.get_V(
                    states1[:, -1]) if step == context.nsteps - 1 else rewards[:, step + 1]
                rewards[:, step] = rewards[:, step] + \
                    (1 - dones[:, step]) * context.gamma * next_r
            mb_states = np.reshape(
                states, [context.minibatch_size] + list(context.env.observation_space.shape))
            mb_rewards = rewards.flatten()
            mb_actions = actions.flatten()
            mb_desired_Q = self.main_brain.get_Q(mb_states)
            all_rows = np.arange(context.minibatch_size)
            mb_error = mb_rewards - mb_desired_Q[all_rows, mb_actions]
            if context.clip_td_error:
                mb_error = np.clip(mb_error, -1, 1)
            mb_desired_Q[all_rows, mb_actions] = mb_desired_Q[all_rows,
                                                              mb_actions] + mb_error
            _, loss, mpe = self.main_brain.train(mb_states, mb_desired_Q)
        if not context.ddpg_target_network_update_mode:
            if context.frame_id % context.target_network_update_interval == 0:
                context.session.run(self.target_brain_update_op)
                logger.log('Target network updated')
        else:
            context.session.run(self.target_brain_soft_update_op)

    def post_episode(self):
        if self.context.episode_id % self.context.save_every == 0:
            filename = self.name.replace('/', '_') + '_model'
            self.main_brain.save(filename=filename)
            self.main_brain.save(filename=filename + str(self.context.episode_id))
            logger.log(filename, "saved", level=logger.DEBUG)

    def close(self):
        filename = self.name.replace('/', '_') + '_model'
        self.main_brain.save(filename=filename)
        self.main_brain.save(filename=filename + str(self.context.episode_id))
        logger.log(filename, "saved", level=logger.DEBUG)


class DQNEvalAgent(Agent):
    def __init__(self, context: Context, name):
        super().__init__(context, name)
        self.main_brain = Brain(context, '{0}/main_brain'.format(name), False)

    def start(self):
        if self.context.load_model_dir is not None:
            self.main_brain.load(
                filename=self.name.replace('/', '_') + '_model')

    def act(self):
        r = np.random.random()
        if r > self.context.epsilon:
            action = self.main_brain.get_action([self.context.frame_obs])[0]
        else:
            action = self.context.env.action_space.sample()
        return action


class DQNSensitivityVisualizerAgent(Agent):
    def __init__(self, context: Context, name):
        super().__init__(context, name)
        self.dqn_agent = None  # type: DQNAgent
        self.window = None

    def start(self):
        if self.dqn_agent is None:
            dqn_agents = self.runner.get_agent_by_type(DQNAgent) + self.runner.get_agent_by_type(DQNEvalAgent)
            assert len(
                dqn_agents) > 0, "There is no DQNAgent registered to this runner"
            self.dqn_agent = dqn_agents[0]
        try:
            if self.context.render:
                self.window = ImagePygletWingow(
                    caption=self.context.env_id + ":" + self.context.experiment_name + ":" + self.name, vsync=self.context.render_vsync)
        except Exception as e:
            logger.error(
                "{0}: Could not create window. Reason = {1}".format(self.name, str(e)))

    def render(self):
        context = self.context
        if self.window and context.need_conv_net and context.render and context.episode_id % context.render_interval == 0:
            frame = self.dqn_agent.main_brain.get_Q_input_sensitivity([context.frame_obs])[0]
            channels = frame.shape[2]
            frame = np.dot(frame.astype('float32'), np.ones([channels]) / channels)
            frame = np.expand_dims(frame, 2)
            frame = np.concatenate([frame] * 3, axis=2)
            frame = (frame * 255).astype(np.uint8)
            orig = context.env.render(mode='rgb_array')
            h, w = orig.shape[0], orig.shape[1]
            frame = np.asarray(Image.fromarray(
                frame).resize((w, h), resample=Image.BILINEAR))
            mixed = 0.9 * frame + 0.1 * orig
            self.window.set_image(mixed.astype(np.uint8))

    def pre_episode(self):
        self.render()

    def post_act(self):
        self.render()

    def close(self):
        if self.window:
            self.window.close()


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
                        sensitivity = main_brain.get_Q_input_sensitivity([obs])[
                            0]
                        sensitivity = (np.sqrt(color_to_grayscale(
                            sensitivity)) * 255).astype(np.uint8)
                        orig = env.render(mode='rgb_array')
                        h, w = orig.shape[0], orig.shape[1]
                        sensitivity = np.asarray(Image.fromarray(
                            sensitivity).resize((w, h), resample=Image.BILINEAR))
                        mixed = 0.8 * sensitivity + 0.2 * orig
                        viewer.imshow(mixed.astype(np.uint8))
                    else:
                        '''In non atari envs, gym monitor calls rendering automatically'''
                        # viewer.imshow(env.render(mode='rgb_array'))

                # let's train:
                if frame_id % context.train_every == 0 and frame_id > context.minimum_experience:
                    states, actions, rewards, dones, infos, states1 = experience_buffer.random_rollouts_unzipped(
                        int(context.minibatch_size / context.nsteps), context.nsteps)
                    for step in range(context.nsteps - 1, -1, -1):
                        next_r = target_brain.get_V(
                            states1[:, -1]) if step == context.nsteps - 1 else rewards[:, step + 1]
                        rewards[:, step] = rewards[:, step] + \
                            (1 - dones[:, step]) * context.gamma * next_r
                    mb_states = np.reshape(
                        states, [context.minibatch_size] + list(env.observation_space.shape))
                    mb_rewards = rewards.flatten()
                    mb_actions = actions.flatten()
                    mb_desired_Q = main_brain.get_Q(mb_states)
                    all_rows = np.arange(context.minibatch_size)
                    mb_error = mb_rewards - mb_desired_Q[all_rows, mb_actions]
                    if context.clip_td_error:
                        mb_error = np.clip(mb_error, -1, 1)
                    mb_desired_Q[all_rows, mb_actions] = mb_desired_Q[all_rows,
                                                                      mb_actions] + mb_error
                    # for exp_id in range(context.minibatch_size):
                    #     error = mb_rewards[exp_id] - mb_desired_Q[exp_id, mb_actions[exp_id]]
                    #     if context.clip_td_error:
                    #         error = np.clip(error, -1, 1)
                    #     mb_desired_Q[exp_id, mb_actions[exp_id]] = mb_desired_Q[exp_id, mb_actions[exp_id]] + error
                    _, loss, mpe = main_brain.train(mb_states, mb_desired_Q)
                    # mb_states1 = np.reshape(states1, [context.minibatch_size] + list(env.observation_space.shape))
                    # half_mb_size = int(context.minibatch_size / 2)
                    # mb_exps_random = list(experience_buffer.random_experiences(half_mb_size))
                    # mb_next_states_random = [exp.next_state for exp in mb_exps_random]
                    # mb_desired_tc = np.zeros([context.minibatch_size, 2])
                    # mb_desired_tc[0:half_mb_size, 0] = 1
                    # mb_desired_tc[half_mb_size:context.minibatch_size, 1] = 1
                    # mb_next_states = mb_next_states_random + list(mb_states1[0:half_mb_size])
                    # mb_actions_onehot = np.zeros([context.minibatch_size, context.env.action_space.n])
                    # mb_actions_onehot[np.arange(context.minibatch_size), mb_actions] = 1
                    # assert len(mb_states) == len(mb_next_states), "{0}, {1}".format(len(mb_states), len(mb_next_states))
                    # main_brain.train_transitions(mb_states, mb_actions_onehot, mb_next_states, mb_desired_tc)
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


class MyContext(Context):
    def wrappers(self, env):
        if self.need_conv_net:
            env = wrap_atari(env, episode_life=self.episode_life, clip_rewards=self.clip_rewards, framestack_k=self.framestack_k)
        if 'Lunar' in self.env_id:
            env = MaxEpisodeStepsWrapper(env, 600)
        return env


if __name__ == '__main__':
    context = MyContext()

    runner = RLRunner(context)

    runner.register_agent(SeedingAgent(context, "seeder"))
    # runner.register_agent(RandomPlayAgent(context, "RandomPlayer"))
    # runner.register_agent(SimpleRenderingAgent(context, "Video"))
    if context.eval_mode:
        runner.register_agent(DQNEvalAgent(context, "DQN"))
    else:
        runner.register_agent(DQNAgent(context, "DQN"))
    if context.need_conv_net:
        runner.register_agent(DQNSensitivityVisualizerAgent(context, "Sensitivity"))
        pass
    runner.register_agent(PygletLoop(context, "PygletLoop"))
    runner.run()

    context.close()
