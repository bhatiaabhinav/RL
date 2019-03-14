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

import os
import gym  # noqa: F401
import numpy as np
from PIL import Image

from RL.common import logger
from RL.common.atari_wrappers import wrap_atari
from RL.common.context import (Agent, Context, PygletLoop, RLRunner,
                               SeedingAgent, SimpleRenderingAgent)
from RL.common.experience_buffer import Experience, MultiRewardStreamExperience, ExperienceBuffer
from RL.common.plot_renderer import PlotRenderer
from RL.common.utils import ImagePygletWingow
from RL.common.wrappers import MaxEpisodeStepsWrapper
from RL.dqn.model import Brain
from typing import List  # noqa: F401


class DQNAgent(Agent):
    def __init__(self, context: Context, name, head_names=["default"], loss_coeffs_per_head=[1.0]):
        super().__init__(context, name)
        self.head_names = head_names
        self.loss_coeffs = loss_coeffs_per_head
        if not hasattr(self.context.gamma, "__len__") and len(head_names) > 1:
            logger.warn("You are using same gamma for all reward streams. Are you sure this is intentional?")
        self.create_main_and_target_brains()
        self.create_experience_buffer()

    def create_main_and_target_brains(self):
        self.main_brain = Brain(
            self.context, '{0}/main_brain'.format(self.name), False, head_names=self.head_names)
        self.target_brain = Brain(
            self.context, '{0}/target_brain'.format(self.name), True, head_names=self.head_names)
        self.target_brain_update_op = self.main_brain.tf_copy_to(
            self.target_brain, soft_copy=False, name='{0}/target_brain_update_op'.format(self.name))
        self.target_brain_soft_update_op = self.main_brain.tf_copy_to(
            self.target_brain, soft_copy=True, name='{0}/target_brain_soft_update_op'.format(self.name))

    def create_experience_buffer(self):
        if self.context.experience_buffer_megabytes is None:
            self.experience_buffer = ExperienceBuffer(
                length=self.context.experience_buffer_length)
        else:
            self.experience_buffer = ExperienceBuffer(
                size_in_bytes=self.context.experience_buffer_megabytes * (1024**2))
        self.nstep_buffer = []  # type: List[Experience]

    def start(self):
        if self.context.load_model_dir is not None:
            self.main_brain.load(
                filename=self.name.replace('/', '_') + '_model')
        self.update_target_network(soft_update=False)

    def pre_act(self):
        self.main_brain.update_running_stats([self.context.frame_obs])

    def exploit_policy(self, states, target_brain=False):
        if not hasattr(self, "_exp_pol_called") and len(self.head_names) > 1:
            logger.log("There are multiple Q heads in agent {0}. Exploit policy will choose greedy action using first head only".format(self.name))
            self._exp_pol_called = True
        brain = self.target_brain if target_brain else self.main_brain
        greedy_actions, = brain.get_argmax_Q(states, head_names=[self.head_names[0]])
        return greedy_actions

    def policy(self, states, target_brain=False):
        r = np.random.random(size=[len(states)])
        exploit_mask = (r > self.context.epsilon).astype(np.int)
        exploit_actions = self.exploit_policy(states, target_brain=target_brain)
        explore_actions = np.random.randint(0, self.context.env.action_space.n, size=[len(states)])
        actions = (1 - exploit_mask) * explore_actions + exploit_mask * exploit_actions
        return actions

    def act(self):
        return self.policy([self.context.frame_obs])[0]

    def add_to_experience_buffer(self, exp: MultiRewardStreamExperience):
        reward_to_prop_back = exp.reward
        for old_exp in reversed(self.nstep_buffer):
            if old_exp.done:
                break
            reward_to_prop_back = np.asarray(self.context.gamma) * reward_to_prop_back
            old_exp.reward += reward_to_prop_back
            old_exp.next_state = exp.next_state
            old_exp.done = exp.done
        self.nstep_buffer.append(exp)
        if len(self.nstep_buffer) >= self.context.nsteps:
            self.experience_buffer.add(self.nstep_buffer.pop(0))
            assert len(self.nstep_buffer) == self.context.nsteps - 1

    def optimize(self, states, actions, *desired_Q_values_per_head):
        all_rows = np.arange(len(states))
        Q_per_head = self.main_brain.get_Q(states)
        for Q, desired_Q_values in zip(Q_per_head, desired_Q_values_per_head):
            td_errors = desired_Q_values - Q[all_rows, actions]
            if self.context.clip_td_error:
                td_errors = np.clip(td_errors, -1, 1)
            Q[all_rows, actions] = Q[all_rows, actions] + td_errors
        combined_loss, Q_losses, Q_mpes = self.main_brain.train(states, *Q_per_head, loss_coeffs_per_head=self.loss_coeffs)
        return combined_loss, Q_losses, Q_mpes

    def get_target_network_V_per_head(self, states):
        all_rows = np.arange(self.context.minibatch_size)
        actions = self.exploit_policy(states, target_brain=not self.context.double_dqn)
        Q_per_head = self.target_brain.get_Q(states)
        Vs = [Q[all_rows, actions] for Q in Q_per_head]
        return Vs

    def train(self):
        c = self.context
        states, actions, multistream_rewards, dones, infos, next_states = self.experience_buffer.random_experiences_unzipped(
            c.minibatch_size)
        next_states_V_per_head = self.get_target_network_V_per_head(next_states)
        desired_Q_values_per_head = []
        gamma = c.gamma
        if not hasattr(c.gamma, "__len__"):
            gamma = [gamma] * len(self.head_names)
        for head_id in range(len(self.head_names)):
            desired_Q_values = multistream_rewards[:, head_id] + (1 - dones.astype(np.int)) * (gamma[head_id] ** c.nsteps) * next_states_V_per_head[head_id]
            desired_Q_values_per_head.append(desired_Q_values)
        combined_loss, Q_losses, Q_mpes = self.optimize(states, actions, *desired_Q_values_per_head)

    def update_target_network(self, soft_update):
        if soft_update:
            self.context.session.run(self.target_brain_soft_update_op)
        else:
            self.context.session.run(self.target_brain_update_op)

    def get_all_stream_rewards_current_frame(self):
        return [self.context.frame_reward]

    def post_act(self):
        c = self.context

        if c.eval_mode:
            return

        reward = self.get_all_stream_rewards_current_frame()
        self.add_to_experience_buffer(MultiRewardStreamExperience(
            c.frame_obs, c.frame_action, reward, c.frame_done, c.frame_info, c.frame_obs_next))

        if c.frame_id % c.train_every == 0 and c.frame_id >= c.minimum_experience:
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
            if self.context.sensitivity_visualizer:
                self.window = ImagePygletWingow(
                    caption=self.context.env_id + ":" + self.context.experiment_name + ":" + self.name, vsync=self.context.render_vsync)
        except Exception as e:
            logger.error(
                "{0}: Could not create window. Reason = {1}".format(self.name, str(e)))

    def render(self):
        context = self.context
        if self.window and context.need_conv_net and context.sensitivity_visualizer and context.episode_id % context.render_interval == 0:
            frame = self.dqn_agent.main_brain.get_Q_input_sensitivity([context.frame_obs])[
                0]
            channels = frame.shape[2]
            frame = np.dot(frame.astype('float32'),
                           np.ones([channels]) / channels)
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


class QPlotAgent(Agent):
    def __init__(self, context: Context, name, reference_mark=0, head_id=0):
        super().__init__(context, name)
        self.dqn_agent = None  # type: DQNAgent
        self.plot = None  # type: ImagePygletWingow
        self.reference_mark = reference_mark
        self.head_id = head_id

    def start(self):
        if self.dqn_agent is None:
            dqn_agents = self.runner.get_agent_by_type(DQNAgent)
            assert len(
                dqn_agents) > 0, "There is no DQNAgent registered to this runner"
            self.dqn_agent = dqn_agents[0]
        try:
            if self.context.plot_Q:
                self.plot = PlotRenderer(title="Q Values ({0}) per Step for ".format(self.dqn_agent.head_names[self.head_id]) + self.dqn_agent.name, xlabel="Step", ylabel="Q", legend=['Q_max', 'Reference', 'Q_min'], window_caption=self.context.env_id + ":" + self.context.experiment_name + ":" + self.name, concat_title_with_caption=False, auto_dispatch_on_render=False)
                self.plot.plot([], [], 'g', [], [], 'k--', [], [], 'r')
        except Exception as e:
            logger.error(
                "{0}: Could not create plot window. Reason = {1}".format(self.name, str(e)))

    def update(self):
        if self.plot and self.context.plot_Q and self.context.episode_id % self.context.render_interval == 0:
            Q_vals, = self.dqn_agent.main_brain.get_Q([self.context.frame_obs], head_names=[self.dqn_agent.head_names[self.head_id]])
            self.plot.append_and_render(
                [np.max(Q_vals), self.reference_mark, np.min(Q_vals)])

    def pre_episode(self):
        if self.plot:
            self.plot.clear()
            self.plot.axes.set_title("Q Values ({0}) per Step for ".format(self.dqn_agent.head_names[self.head_id]) + self.dqn_agent.name + " : Episode " + str(self.context.episode_id))
            self.update()

    def post_episode(self):
        if self.plot and self.context.plot_Q and self.context.episode_id % self.context.render_interval == 0:
            filename = os.path.join(logger.get_dir(
            ), "Plots", self.name, "Q_Episode_{0}.png".format(self.context.episode_id))
            self.plot.save(path=filename)

    def post_act(self):
        self.update()

    def close(self):
        if self.plot:
            self.plot.close()


class MyContext(Context):
    def wrappers(self, env):
        if self.need_conv_net:
            env = wrap_atari(env, episode_life=self.atari_episode_life,
                             clip_rewards=self.atari_clip_rewards, framestack_k=self.atari_framestack_k)
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
    if context.plot_Q:
        runner.register_agent(QPlotAgent(context, "DQN_Q", head_id=0))
    runner.register_agent(PygletLoop(context, "PygletLoop"))
    runner.run()

    context.close()
