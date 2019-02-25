"""dqn.py: Implementation of Deepmind's DQN algorithm using Tensorflow and OpenAI gym.
extras:
    Double DQN
    Dueling DQN
    Observation Normalization
    n-step learning
    DDPG style soft updated target network
    clipping, regularization etc.
    TODO: RND prediction based (episodic) intrinsic reward (single value head) - a naive implementation

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
from PIL import Image

from RL.common import logger
from RL.common.atari_wrappers import wrap_atari
from RL.common.context import (Agent, Context, PygletLoop, RLRunner,
                               SeedingAgent, SimpleRenderingAgent)
from RL.common.experience_buffer import Experience, ExperienceBuffer
from RL.common.utils import ImagePygletWingow
from RL.common.wrappers import MaxEpisodeStepsWrapper
from RL.dqn.model import Brain
from typing import List  # noqa: F401


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
        self.nstep_buffer = []  # type: List[Experience]

    def start(self):
        if self.context.load_model_dir is not None:
            self.main_brain.load(
                filename=self.name.replace('/', '_') + '_model')
        self.update_target_network(soft_update=False)

    def pre_act(self):
        self.main_brain.update_running_stats([self.context.frame_obs])

    def act(self):
        r = np.random.random()
        if r > self.context.epsilon:
            action = self.main_brain.get_action([self.context.frame_obs])[0]
        else:
            action = self.context.env.action_space.sample()
        return action

    def add_to_experience_buffer(self, exp: Experience):
        reward_to_prop_back = exp.reward
        for old_exp in reversed(self.nstep_buffer):
            if old_exp.done:
                break
            reward_to_prop_back = self.context.gamma * reward_to_prop_back
            old_exp.reward += reward_to_prop_back
            old_exp.next_state = exp.next_state
            old_exp.done = exp.done
        self.nstep_buffer.append(exp)
        if len(self.nstep_buffer) >= self.context.nsteps:
            self.experience_buffer.add(self.nstep_buffer.pop(0))
            assert len(self.nstep_buffer) == self.context.nsteps - 1

    def optimize(self, states, actions, desired_Q_values):
        all_rows = np.arange(self.context.minibatch_size)
        Q = self.main_brain.get_Q(states)
        td_errors = desired_Q_values - Q[all_rows, actions]
        if self.context.clip_td_error:
            td_errors = np.clip(td_errors, -1, 1)
        Q[all_rows, actions] = Q[all_rows, actions] + td_errors
        _, loss, mpe = self.main_brain.train(states, Q)
        return loss, mpe

    def get_target_network_V(self, states):
        all_rows = np.arange(self.context.minibatch_size)
        if self.context.double_dqn:
            V = self.target_brain.get_Q(
                states)[all_rows, self.main_brain.get_action(states)]
        else:
            V = self.target_brain.get_V(states)
        return V

    def train(self):
        c = self.context
        states, actions, rewards, dones, infos, next_states = self.experience_buffer.random_experiences_unzipped(
            c.minibatch_size)
        next_states_V = self.get_target_network_V(next_states)
        desired_Q_values = rewards + (1 - dones.astype(np.int)) * (c.gamma ** c.nsteps) * next_states_V
        loss, mpe = self.optimize(states, actions, desired_Q_values)

    def update_target_network(self, soft_update):
        if soft_update:
            self.context.session.run(self.target_brain_soft_update_op)
        else:
            self.context.session.run(self.target_brain_update_op)

    def post_act(self):
        c = self.context

        if c.eval_mode:
            return

        self.add_to_experience_buffer(Experience(c.frame_obs, c.frame_action, c.frame_reward, c.frame_done, c.frame_info, c.frame_obs_next))

        if c.frame_id % c.train_every == 0 and c.frame_id > c.minimum_experience:
            self.train()

        if c.frame_id % c.target_network_update_every == 0:
            self.update_target_network(soft_update=True)

    def save_model(self):
        filename = self.name.replace('/', '_') + '_model'
        self.main_brain.save(filename=filename)
        self.main_brain.save(filename=filename + str(self.context.episode_id))
        logger.log(filename, "saved", level=logger.DEBUG)

    def post_episode(self):
        if self.context.eval_mode:
            return
        if self.context.episode_id % self.context.save_every == 0:
            self.save_model()

    def close(self):
        if self.context.eval_mode:
            return
        self.save_model()


class DQNSensitivityVisualizerAgent(Agent):
    def __init__(self, context: Context, name):
        super().__init__(context, name)
        self.dqn_agent = None  # type: DQNAgent
        self.window = None

    def start(self):
        if self.dqn_agent is None:
            dqn_agents = self.runner.get_agent_by_type(DQNAgent)
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


class MyContext(Context):
    def wrappers(self, env):
        if self.need_conv_net:
            env = wrap_atari(env, episode_life=self.episode_life,
                             clip_rewards=self.clip_rewards, framestack_k=self.framestack_k)
        if 'Lunar' in self.env_id:
            env = MaxEpisodeStepsWrapper(env, 600)
        return env


if __name__ == '__main__':
    context = MyContext()

    runner = RLRunner(context)

    runner.register_agent(SeedingAgent(context, "seeder"))
    # runner.register_agent(RandomPlayAgent(context, "RandomPlayer"))
    if context.render:
        runner.register_agent(SimpleRenderingAgent(context, "Video"))
    runner.register_agent(DQNAgent(context, "DQN"))
    if context.sensitivity_visualizer:
        runner.register_agent(
            DQNSensitivityVisualizerAgent(context, "Sensitivity"))
    runner.register_agent(PygletLoop(context, "PygletLoop"))
    runner.run()

    context.close()
