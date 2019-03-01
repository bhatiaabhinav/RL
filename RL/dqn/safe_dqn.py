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
    python3 dqn.py [env_id]
    # env_id can be any environment with discrete action space.
    env_id: optional. default='CartPole-v0'. refer https://gym.openai.com/envs/

python dependencies:
    gym[classic_control], tensorflow, numpy
"""

__author__ = "Abhinav Bhatia"
__email__ = "bhatiaabhinav93@gmail.com"
__license__ = "gpl"
__version__ = "1.0.0"

from collections import deque
from typing import List  # noqa: F401

import gym  # noqa: F401
import numpy as np

from RL.common import logger
from RL.common.atari_wrappers import wrap_atari
from RL.common.context import (
    Agent, Context, PygletLoop, RLRunner, SeedingAgent, SimpleRenderingAgent)
from RL.common.utils import ImagePygletWingow
from RL.common.wrappers import MaxEpisodeStepsWrapper
from RL.dqn.dqn import DQNAgent, DQNSensitivityVisualizerAgent, QPlotAgent  # noqa: F401


class SafetyDQNAgent(DQNAgent):
    def __init__(self, context: Context, name):
        super().__init__(context, name)
        self.reward_key = "{0}_reward".format(name)
        self.safety_threshold = context.safety_threshold
        self.min_threshold = self.safety_threshold
        self.max_threshold = 0.01
        logger.log("safety thres is ", str(self.safety_threshold))

    def act(self):
        return None

    def get_feasibility_mask_and_Q(self, states, target_brain=False):
        if target_brain:
            Q = self.target_brain.get_Q(states)
        else:
            Q = self.main_brain.get_Q(states)
        t = self.threshold_logic(Q)
        if not target_brain:
            self.safety_threshold = t
        mask = (Q >= t).astype(np.int)
        return mask, Q

    def threshold_logic(self, Q):
        max_Q = np.max(Q, axis=-1, keepdims=True)
        min_Q = np.min(Q, axis=-1, keepdims=True)
        range_Q = max_Q - min_Q
        thres = max_Q - 0.5 * range_Q
        thres = np.clip(thres, self.min_threshold, self.max_threshold)
        return thres

    def post_act(self):
        actual_reward = self.context.frame_reward
        assert self.reward_key in self.context.frame_info, "reward for safety agent '{0}' is missing from info of the latest step. Make sure the info dictionary has a key called {1} specifying a non positive reward".format(
            self.name, self.reward_key)
        safety_reward = self.context.frame_info[self.reward_key]
        assert safety_reward <= 0, "{0} should not be positive".format(
            self.reward_key)
        self.context.frame_reward = safety_reward
        super().post_act()
        self.context.frame_reward = actual_reward


class SafetyAwareDQNAgent(DQNAgent):
    def start(self):
        super().start()
        self.safety_agents = self.runner.get_agent_by_type(
            SafetyDQNAgent)  # type: List[SafetyDQNAgent]
        assert len(self.safety_agents) > 0, "No safety agent found!"
        logger.info("Safety Aware DQN found {0} safety agents".format(
            len(self.safety_agents)))

    def _reset_feasible_actions_count_stats(self):
        self.av_feasible_actions_count = 0
        self.nonrandom_actions_count = 0

    def _update_feasible_actions_count_stats(self, feasible_action_count):
        self.av_feasible_actions_count = (self.av_feasible_actions_count * self.nonrandom_actions_count + feasible_action_count) / (self.nonrandom_actions_count + 1)
        self.nonrandom_actions_count += 1

    def pre_episode(self):
        super().pre_episode()
        self._reset_feasible_actions_count_stats()

    def get_combined_safety_feasibility_mask_and_Q(self, states, target_brain=False):
        combined_mask, combined_Q = 1, 0
        for safety_agent in self.safety_agents:
            mask, Q = safety_agent.get_feasibility_mask_and_Q(states, target_brain=target_brain)
            combined_mask = combined_mask * mask
            combined_Q = combined_Q + Q
        return combined_mask, combined_Q

    def get_feasibleGreedy_action_and_Q(self, states, feasibility_mask, safety_Q, target_brain=False):
        '''feasibleGreedy policy is defined as: Choose the action with highest Q value among feasible action. If there is no feasible action, choose the safest action'''

        if target_brain:
            Q = self.target_brain.get_Q(states)
        else:
            Q = self.main_brain.get_Q(states)
        feasible_available_mask = np.any(feasibility_mask, axis=-1).astype(np.int)
        # additive mask should have -inf for infeasible actions and 0 for feasible:
        additive_mask = (1 - feasibility_mask) * -1e6
        Q_masked = Q + additive_mask
        action_feasible = np.argmax(Q_masked, axis=-1)
        action_safest = np.argmax(safety_Q, axis=-1)
        action = feasible_available_mask * action_feasible + (1 - feasible_available_mask) * action_safest
        return action, Q

    def get_target_network_V(self, states):
        all_rows = np.arange(self.context.minibatch_size)
        feasibility_mask, safety_Q = self.get_combined_safety_feasibility_mask_and_Q(states, target_brain=True)
        feasibleGreedy_action, Q = self.get_feasibleGreedy_action_and_Q(states, feasibility_mask, safety_Q, target_brain=True)
        if self.context.double_dqn:
            feasibleGreedy_action_main, Q_main = self.get_feasibleGreedy_action_and_Q(states, feasibility_mask, safety_Q, target_brain=False)
            return Q[all_rows, feasibleGreedy_action_main]
        else:
            return Q[all_rows, feasibleGreedy_action]

    def act(self):
        r = np.random.random()
        if r > self.context.epsilon:
            feasibility_mask, safety_Q = self.get_combined_safety_feasibility_mask_and_Q([self.context.frame_obs])
            feasibleGreedy_action, Q = self.get_feasibleGreedy_action_and_Q([self.context.frame_obs], feasibility_mask, safety_Q)
            action = feasibleGreedy_action[0]
            self._update_feasible_actions_count_stats(np.sum(feasibility_mask[0]))
        else:
            action = self.context.env.action_space.sample()
        return action

    def post_episode(self):
        super().post_episode()
        av_feasible_actions_count_summary_name = "av_feasible_actions_count"
        if self.context.episode_id == 0:
            self.context.summaries.setup_scalar_summaries(
                [av_feasible_actions_count_summary_name])
        self.context.log_summary(
            {av_feasible_actions_count_summary_name: self.av_feasible_actions_count}, self.context.episode_id)


class PenaltyBasedSafeDQN(DQNAgent):
    def __init__(self, context: Context, name):
        super().__init__(context, name)
        self.safety_reward_keys = [name + "_reward" for name in context.safety_stream_names]

    def post_act(self):
        total_safety_reward = self.context.penalty_safe_dqn_multiplier * \
            sum([self.context.frame_info.get(k, 0)
                 for k in self.safety_reward_keys])
        self.context.frame_reward += total_safety_reward
        super().post_act()
        self.context.frame_reward -= total_safety_reward


class LunarLanderSafetyWrapper(gym.Wrapper):
    def __init__(self, env, no_crash_penalty_main_stream=False):
        super().__init__(env)
        self.no_crash_penalty_main_stream = no_crash_penalty_main_stream

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs_next, reward, done, info = self.env.step(action)
        if done and reward == -100:
            '''it is a crash'''
            info["Safety_reward"] = -1 + info.get('Safety_reward', 0)
            if self.no_crash_penalty_main_stream:
                reward = 0
        else:
            info["Safety_reward"] = 0 + info.get('Safety_reward', 0)
        return obs_next, reward, done, info


class PongSafetyWrapper(gym.Wrapper):
    def __init__(self, env, render=False):
        super().__init__(env)
        self.window = None
        self.ball_lefts = deque([], 4)
        self.strike_no = 0
        if render:
            try:
                self.window = ImagePygletWingow(caption="Ball and bat")
            except Exception as e:
                logger.error(
                    "RenderWrapper: Could not create window. Reason = {0}".format(str(e)))

    def get_safety_reward(self, obs):
        frame = ((np.dot(obs.astype('float32'), np.array(
            [0.299, 0.587, 0.114], 'float32')) > 128) * 255).astype(np.uint8)
        frame = np.concatenate([np.expand_dims(frame, 2)] * 3, axis=2)
        frame = self.bat_position(frame)
        frame = self.ball_position(frame)
        safety_reward = 0
        if len(self.ball_lefts) == 4:
            vel1 = self.ball_lefts[1] - self.ball_lefts[0]
            vel2 = self.ball_lefts[3] - self.ball_lefts[2]
            if vel1 > 0 and vel2 < 0 and max(self.ball_lefts) - min(self.ball_lefts) < 10:
                self.strike_no += 1
                # print("Strike {0}. q={1}".format(self.strike_no, self.ball_lefts))
                self.ball_lefts.clear()
                ball_mid = (2 * self.ball_last_seen[0] + 4 - 1) // 2
                bat_mid = (2 * self.bat_last_seen[0] + 16 - 1) // 2
                if ball_mid > bat_mid:
                    # unsafe behavior
                    safety_reward = -1
                    print("Unsafe Strike")
                else:
                    print("Safe Strike")
        if self.window:
            self.window.imshow(frame)
        return safety_reward

    def bat_position(self, frame):
        band = np.sum(frame[34:-16, 140:144, 0],
                      axis=-1, dtype=np.int) > (3 * 255)
        nonzero = band.nonzero()[0]
        if len(nonzero) > 0:
            if band[0] == 255:
                bottom = 34 + nonzero[-1]
                top = bottom - 16 + 1
            else:
                top = 34 + nonzero[0]
                bottom = top + 16 - 1
            # mid = (top + bottom) // 2
            frame[top: top + 16, 140:144, 0] = 0
            frame[top: top + 16, 140:144, 1] = 0
            frame[top: top + 16, 140:144, 2] = 255
            self.bat_last_seen = [top, 140]
        return frame

    def ball_position(self, frame):
        search_region = frame[34:-16, 30:, 0].astype(np.float32) / 255
        nonzero = np.nonzero(search_region)
        if len(nonzero[0]) > 0 and len(nonzero[1]) > 0:
            top = nonzero[0][0]
            left = nonzero[1][0]
            top += 34
            left += 30
            frame[top:top + 4, left:left + 2, 0] = 0
            frame[top:top + 4, left:left + 2, 1] = 255
            frame[top:top + 4, left:left + 2, 2] = 0
            self.ball_last_seen = [top, left]
            self.ball_lefts.append(left)
        return frame

    def reset(self):
        self.ball_lefts.clear()
        self.ball_last_seen = [-1, -1]
        self.bat_last_seen = [-1, -1]
        obs = self.env.reset()
        return obs

    def step(self, action):
        # if self.ball_last_seen[0] > self.bat_last_seen[0]:
        #     action = 3
        # elif self.ball_last_seen[0] < self.bat_last_seen[0]:
        #     action = 2
        # else:
        #     action = 0
        obs_next, reward, done, info = self.env.step(action)
        safety_reward = self.get_safety_reward(self.env.render('rgb_array'))
        info["Safety_reward"] = safety_reward + info.get("Safety_reward", 0)
        return obs_next, reward, done, info

    def close(self):
        if self.window:
            self.window.close()


class SafetyStatsRecorderAgent(Agent):
    def start(self):
        assert len(self.runner.get_agent_by_type(SafetyStatsRecorderAgent)
                   ) == 1, "There cannot be more than 1 SafetyStatsRecorder in a runner"
        self.safety_agent_names = self.context.safety_stream_names
        self.safety_keys = [name + "_reward" for name in self.safety_agent_names]
        self.safety_records = {}
        self.safety_records_exploit = {}
        self.safety_R = {}
        for name in self.safety_agent_names:
            self.safety_records[name] = []
            self.safety_records_exploit[name] = []
            self.safety_R[name] = 0

    def pre_episode(self):
        for name in self.safety_agent_names:
            self.safety_R[name] = 0

    def post_act(self):
        for name, key in zip(self.safety_agent_names, self.safety_keys):
            self.safety_R[name] += self.context.frame_info.get(key, 0)

    def post_episode(self):
        summary = {}
        for name, key in zip(self.safety_agent_names, self.safety_keys):
            self.safety_records[name].append(self.safety_R[name])
            R_summary_name = name + "_R"
            R_exploit_summary_name = name + "_R_exploit"
            cumulative_R_summary_name = name + "_cumulative_R"
            cumulative_R_exploit_summary_name = name + "_cumulative_R_exploit"
            if self.context.episode_id == 0:
                self.context.summaries.setup_scalar_summaries(
                    [R_summary_name, R_exploit_summary_name, cumulative_R_summary_name, cumulative_R_exploit_summary_name])
            summary[R_summary_name] = self.safety_R[name]
            summary[cumulative_R_summary_name] = sum(
                self.safety_records[name])
            if self.context.should_eval_episode():
                self.safety_records_exploit[name].append(
                    self.safety_R[name])
                summary[R_exploit_summary_name] = self.safety_R[name]
                summary[cumulative_R_exploit_summary_name] = sum(
                    self.safety_records_exploit[name])
        self.context.log_summary(summary, self.context.episode_id)


class SafetyQPlotAgent(QPlotAgent):
    def update(self):
        self.reference_mark = getattr(self.dqn_agent, "safety_threshold")
        super().update()


class MyContext(Context):
    def wrappers(self, env):
        if self.need_conv_net:
            if 'Pong' in self.env_id:
                env = PongSafetyWrapper(env)
            env = wrap_atari(env, episode_life=self.atari_episode_life,
                             clip_rewards=self.atari_clip_rewards, framestack_k=self.atari_framestack_k)
        if 'Lunar' in self.env_id:
            env = MaxEpisodeStepsWrapper(env, 600)
            env = LunarLanderSafetyWrapper(env, no_crash_penalty_main_stream=self.lunar_no_crash_penalty_main_stream)
        return env


if __name__ == '__main__':
    context = MyContext()
    runner = RLRunner(context)

    runner.register_agent(SeedingAgent(context, "seeder"))
    # runner.register_agent(RandomPlayAgent(context, "RandomPlayer"))
    if context.render:
        runner.register_agent(SimpleRenderingAgent(context, "Video"))
    # runner.register_agent(DQNAgent(context, "DQN"))
    runner.register_agent(SafetyStatsRecorderAgent(context, "SafetyStats"))
    dqn_agent = None  # type: DQNAgent
    if not context.penalty_safe_dqn_mode:
        for name in context.safety_stream_names:
            safety_dqn_agent = runner.register_agent(SafetyDQNAgent(context, name))
            if context.plot_Q:
                q_plot_agent = runner.register_agent(SafetyQPlotAgent(context, safety_dqn_agent.name + "_Q"))  # type: SafetyQPlotAgent
                q_plot_agent.dqn_agent = safety_dqn_agent
        dqn_agent = runner.register_agent(SafetyAwareDQNAgent(context, "DQN"))
    else:
        dqn_agent = runner.register_agent(PenaltyBasedSafeDQN(context, "DQN"))
    q_plot_agent = runner.register_agent(QPlotAgent(context, dqn_agent.name + "_Q"))  # type: QPlotAgent
    q_plot_agent.dqn_agent = dqn_agent
    if context.sensitivity_visualizer:
        sens_vis_agent = runner.register_agent(DQNSensitivityVisualizerAgent(context, "Sensitivity"))  # type: DQNSensitivityVisualizerAgent
        sens_vis_agent.dqn_agent = dqn_agent
    runner.register_agent(PygletLoop(context, "PygletLoop"))
    runner.run()

    context.close()
