import RL
import numpy as np


class BasicStatsRecordingAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, frameskip=1):
        super().__init__(context, name)
        keys = ['Env-{0} Episode ID', 'Env-{0} Episode Type', 'Env-{0} Episode Length', 'Env-{0} Total Steps', 'Env-{0} Total Frames', 'Env-{0} Total Episodes', 'Env-{0} Episode Timestamp', 'Env-{0} Episode Reward', 'Env-{0} Exploit Episode Reward', 'Env-{0} Av100 Episode Reward', 'Env-{0} Av100 Exploit Episode Reward']
        self.frameskip = frameskip
        for env_id_no in range(self.context.num_envs):
            for k in keys:
                RL.stats.declare_key(k.format(env_id_no), RL.Stats.KeyType.SCALAR if 'Av100' in k else RL.Stats.KeyType.LIST)

    def start(self):
        RL.stats.record_start_time()
        self.episode_rewards = np.zeros(self.context.num_envs)
        self.episode_lengths = np.zeros(self.context.num_envs, dtype=np.int)

    def post_act(self):
        self.episode_rewards = self.episode_rewards + self.runner.rewards
        self.episode_lengths = self.episode_lengths + 1

    def record(self, env_id_nos):
        for env_id_no in env_id_nos:
            RL.stats.record_append('Env-{0} Episode ID'.format(env_id_no), self.runner.episode_ids[env_id_no])
            RL.stats.record_append('Env-{0} Episode Type'.format(env_id_no), 'Exploit' if self.runner.exploit_modes[env_id_no] else 'Train')
            RL.stats.record_append('Env-{0} Episode Length'.format(env_id_no), self.episode_lengths[env_id_no])
            RL.stats.record_append('Env-{0} Total Steps'.format(env_id_no), self.runner.step_ids[env_id_no] + 1)
            RL.stats.record_append('Env-{0} Total Frames'.format(env_id_no), self.frameskip * (self.runner.step_ids[env_id_no] + 1))
            RL.stats.record_append('Env-{0} Total Episodes'.format(env_id_no), self.runner.episode_ids[env_id_no] + 1)
            RL.stats.record_time_append('Env-{0} Episode Timestamp'.format(env_id_no))
            # rewards
            RL.stats.record_append('Env-{0} Episode Reward'.format(env_id_no), self.episode_rewards[env_id_no])
            # exploit rewards
            exploit_rewards = RL.stats.get('Env-{0} Exploit Episode Reward'.format(env_id_no))
            if self.runner.exploit_modes[env_id_no]:
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

    def pre_episode(self, env_id_nos):
        self.episode_rewards[env_id_nos] = 0
        self.episode_lengths[env_id_nos] = 0

    def post_episode(self, env_id_nos):
        self.record(env_id_nos)
