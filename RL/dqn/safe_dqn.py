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
from RL.common.context import (Agent, Context, PygletLoop, RLRunner, SeedingAgent, SimpleRenderingAgent)
from RL.common.utils import ImagePygletWingow
from RL.common.wrappers import MaxEpisodeStepsWrapper
from RL.dqn.dqn import DQNAgent, DQNSensitivityVisualizerAgent  # noqa: F401


class SafetyDQNAgent(DQNAgent):
    def __init__(self, context: Context, name):
        super().__init__(context, name)
        self.reward_key = "{0}_reward".format(name)
        self.safety_threshold = context.safety_threshold
        logger.log("safety thres is", str(self.safety_threshold))

    def act(self):
        return None

    def post_act(self):
        actual_reward = self.context.frame_reward
        assert self.reward_key in self.context.frame_info, "reward for safety agent '{0}' is missing from info of the latest step. Make sure the info dictionary has a key called {1} specifying a non positive reward".format(
            self.name, self.reward_key)
        safety_reward = self.context.frame_info[self.reward_key]
        assert safety_reward <= 0, "{0} should not be positive".format(
            self.reward_key)
        self.context.frame_reward = safety_reward
        actual_gamma = self.context.gamma
        self.context.gamma = 1
        super().post_act()
        self.context.gamma = actual_gamma
        self.context.frame_reward = actual_reward


class SafetyAwareDQNAgent(DQNAgent):
    def start(self):
        super().start()
        self.safety_agents = self.runner.get_agent_by_type(
            SafetyDQNAgent)  # type: List[SafetyDQNAgent]
        assert len(self.safety_agents) > 0, "No safety agent found!"
        logger.info("Safety Aware DQN found {0} safety agents".format(
            len(self.safety_agents)))

    def pre_episode(self):
        super().pre_episode()
        self.av_feasible_actions_count = 0
        self.exploit_actions_count = 0

    def act(self):
        r = np.random.random()
        if r > self.context.epsilon:
            feasible_action_ids = set(range(self.context.env.action_space.n))
            combined_safety_Q_values = np.zeros(
                [self.context.env.action_space.n])
            for safety_agent in self.safety_agents:
                safety_Q_values = safety_agent.main_brain.get_Q(
                    [self.context.frame_obs])[0]
                combined_safety_Q_values = combined_safety_Q_values + safety_Q_values
                feasible_action_ids_ = set(tup[0] for tup in filter(
                    lambda tup: tup[1] >= safety_agent.safety_threshold, zip(feasible_action_ids, safety_Q_values)))
                feasible_action_ids.intersection_update(feasible_action_ids_)
            if len(feasible_action_ids) > 0:
                objective_Q_values = self.main_brain.get_Q(
                    [self.context.frame_obs])[0]
                objective_Q_values_modified = [objective_Q_values[i] if i in feasible_action_ids else -float(
                    "inf") for i in range(self.context.env.action_space.n)]
                action = np.argmax(objective_Q_values_modified)
            else:
                # no feasible action. Let's take the safest one
                action = np.argmax(combined_safety_Q_values)
            self.av_feasible_actions_count = (self.av_feasible_actions_count * self.exploit_actions_count + len(
                feasible_action_ids)) / (self.exploit_actions_count + 1)
            self.exploit_actions_count += 1
        else:
            action = self.context.env.action_space.sample()
        return action

    def post_act(self):
        if any([self.context.frame_info[safety_agent.reward_key] < 0 for safety_agent in self.safety_agents]):
            reward = self.context.frame_reward
            self.context.frame_reward = 0
            super().post_act()
            self.context.frame_reward = reward
        else:
            super().post_act()

    def post_episode(self):
        super().post_episode()
        av_feasible_actions_count_summary_name = "av_feasible_actions_count"
        if self.context.episode_id == 0:
            self.context.summaries.setup_scalar_summaries(
                [av_feasible_actions_count_summary_name])
        self.context.log_summary(
            {av_feasible_actions_count_summary_name: self.av_feasible_actions_count}, self.context.episode_id)


class LunarLanderSafetyWrapper(gym.Wrapper):
    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs_next, reward, done, info = self.env.step(action)
        if done and reward == -100:
            '''it is a crash'''
            info["Safety_reward"] = -1 + info.get('Safety_reward', 0)
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
        self.safety_agents = self.runner.get_agent_by_type(
            SafetyDQNAgent)  # type: List[SafetyDQNAgent]
        self.safety_records = {}
        self.safety_records_exploit = {}
        self.safety_R = {}
        for safety_agent in self.safety_agents:
            self.safety_records[safety_agent.name] = []
            self.safety_records_exploit[safety_agent.name] = []
            self.safety_R[safety_agent.name] = 0

    def pre_episode(self):
        for safety_agent in self.safety_agents:
            self.safety_R[safety_agent.name] = 0

    def post_act(self):
        for safety_agent in self.safety_agents:
            self.safety_R[safety_agent.name] += self.context.frame_info.get(
                safety_agent.reward_key)

    def post_episode(self):
        summary = {}
        for safety_agent in self.safety_agents:
            self.safety_records[safety_agent.name].append(
                self.safety_R[safety_agent.name])
            R_summary_name = safety_agent.name + "_R"
            R_exploit_summary_name = safety_agent.name + "_R_exploit"
            cumulative_R_summary_name = safety_agent.name + "_cumulative_R"
            cumulative_R_exploit_summary_name = safety_agent.name + "_cumulative_R_exploit"
            if self.context.episode_id == 0:
                self.context.summaries.setup_scalar_summaries(
                    [R_summary_name, R_exploit_summary_name, cumulative_R_summary_name, cumulative_R_exploit_summary_name])
            summary[R_summary_name] = self.safety_R[safety_agent.name]
            summary[cumulative_R_summary_name] = sum(
                self.safety_records[safety_agent.name])
            if self.context.should_eval_episode():
                self.safety_records_exploit[safety_agent.name].append(self.safety_R[safety_agent.name])
                summary[R_exploit_summary_name] = self.safety_R[safety_agent.name]
                summary[cumulative_R_exploit_summary_name] = sum(
                    self.safety_records_exploit[safety_agent.name])
        self.context.log_summary(summary, self.context.episode_id)


class MyContext(Context):
    def wrappers(self, env):
        if self.need_conv_net:
            if 'Pong' in self.env_id:
                env = PongSafetyWrapper(env)
            env = wrap_atari(env, episode_life=self.episode_life, clip_rewards=self.clip_rewards, framestack_k=self.framestack_k)
        if 'Lunar' in self.env_id:
            env = MaxEpisodeStepsWrapper(env, 600)
        return env


if __name__ == '__main__':
    context = MyContext()
    runner = RLRunner(context)

    runner.register_agent(SeedingAgent(context, "seeder"))
    # runner.register_agent(RandomPlayAgent(context, "RandomPlayer"))
    runner.register_agent(SimpleRenderingAgent(context, "Video"))
    # runner.register_agent(DQNAgent(context, "DQN"))
    runner.register_agent(SafetyDQNAgent(context, "Safety"))
    runner.register_agent(SafetyStatsRecorderAgent(context, "SafetyStats"))
    runner.register_agent(SafetyAwareDQNAgent(context, "DQN"))
    if context.need_conv_net:
        # runner.register_agent(DQNSensitivityVisualizerAgent(context, "Sensitivity"))
        pass
    runner.register_agent(PygletLoop(context, "PygletLoop"))
    runner.run()

    context.close()
