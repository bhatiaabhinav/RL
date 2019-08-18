import RL
import numpy as np
import time


class BasicStatsRecordingAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, frameskip=1, record_returns=None, record_unscaled_rewards=None, reward_scaling=None):
        super().__init__(context, name)
        keys = ['Env-{0} Episode ID', 'Env-{0} Episode Type', 'Env-{0} Episode Length', 'Env-{0} Total Steps', 'Env-{0} Total Frames', 'Env-{0} Total Episodes', 'Env-{0} Episode Timestamp', 'Env-{0} Episode Reward', 'Env-{0} Exploit Episode Reward', 'Env-{0} Av100 Episode Reward', 'Env-{0} Av100 Exploit Episode Reward', 'Env-{0} Episode Cost', 'Env-{0} Episode FPS']
        self.frameskip = frameskip
        self.record_returns = context.record_returns if record_returns is None else record_returns
        self.record_unscaled_rewards = context.record_unscaled_rewards if record_unscaled_rewards is None else record_unscaled_rewards
        self.reward_scaling = context.reward_scaling if reward_scaling is None else reward_scaling
        self._reward_unscaling = 1 / self.reward_scaling if self.record_unscaled_rewards else 1
        for env_id_no in range(self.context.num_envs):
            for k in keys:
                RL.stats.declare_key(k.format(env_id_no), RL.Stats.KeyType.SCALAR if 'Av100' in k else RL.Stats.KeyType.LIST)

    def start(self):
        RL.stats.record_start_time()
        self.episode_rewards = np.zeros(self.context.num_envs)
        self.episode_safety_rewards = np.zeros(self.context.num_envs)

    def post_act(self):
        if self.record_returns:
            self.episode_rewards += (self.context.gamma ** self.runner.episode_step_ids) * self.runner.rewards * self._reward_unscaling
        else:
            self.episode_rewards += self.runner.rewards * self._reward_unscaling
        safety_rewards = np.asarray([i.get('Safety_reward', 0) for i in self.runner.infos])
        if self.record_returns:
            self.episode_safety_rewards += (self.context.safety_gamma ** self.runner.episode_step_ids) * safety_rewards * self._reward_unscaling
        else:
            self.episode_safety_rewards += safety_rewards * self._reward_unscaling
        self.episode_steps += 1

    def record(self, env_id_nos):
        for env_id_no in env_id_nos:
            RL.stats.record_append('Env-{0} Episode ID'.format(env_id_no), self.runner.episode_ids[env_id_no])
            RL.stats.record_append('Env-{0} Episode Type'.format(env_id_no), 'Exploit' if self.context.force_exploits[env_id_no] else 'Train')
            RL.stats.record_append('Env-{0} Episode Length'.format(env_id_no), self.runner.episode_step_ids[env_id_no] + 1)
            RL.stats.record_append('Env-{0} Total Steps'.format(env_id_no), self.runner.step_ids[env_id_no] + 1)
            RL.stats.record_append('Env-{0} Total Frames'.format(env_id_no), self.frameskip * (self.runner.step_ids[env_id_no] + 1))
            RL.stats.record_append('Env-{0} Total Episodes'.format(env_id_no), self.runner.episode_ids[env_id_no] + 1)
            RL.stats.record_time_append('Env-{0} Episode Timestamp'.format(env_id_no))
            # rewards
            RL.stats.record_append('Env-{0} Episode Reward'.format(env_id_no), self.episode_rewards[env_id_no])
            RL.stats.record_append('Env-{0} Episode Cost'.format(env_id_no), -self.episode_safety_rewards[env_id_no])
            # exploit rewards
            exploit_rewards = RL.stats.get('Env-{0} Exploit Episode Reward'.format(env_id_no))
            if self.context.force_exploits[env_id_no]:
                RL.stats.record_append('Env-{0} Exploit Episode Reward'.format(env_id_no), self.episode_rewards[env_id_no])
            elif exploit_rewards is not None and len(exploit_rewards) > 0:
                RL.stats.record_append('Env-{0} Exploit Episode Reward'.format(env_id_no), exploit_rewards[-1])
            else:
                RL.stats.record_append('Env-{0} Exploit Episode Reward'.format(env_id_no), 0)
            # mean rewards:
            rewards = RL.stats.get('Env-{0} Episode Reward'.format(env_id_no))
            exploit_rewards = RL.stats.get('Env-{0} Exploit Episode Reward'.format(env_id_no))
            av_rpe = np.mean(rewards[-100:])
            av_rpe_exploit = np.mean(exploit_rewards[-100:])
            RL.stats.record('Env-{0} Av100 Episode Reward'.format(env_id_no), av_rpe)
            RL.stats.record('Env-{0} Av100 Exploit Episode Reward'.format(env_id_no), av_rpe_exploit)
            RL.stats.record_append('Env-{0} Episode FPS'.format(env_id_no), self.episode_fps)

    def pre_episode(self, env_id_nos):
        self.episode_rewards[env_id_nos] = 0
        self.episode_safety_rewards[env_id_nos] = 0
        self.episode_steps = 0
        self.ep_start_time = time.time()

    def post_episode(self, env_id_nos):
        self.episode_fps = self.episode_steps / (time.time() - self.ep_start_time)
        self.record(env_id_nos)
