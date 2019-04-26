import RL
import numpy as np


class BasicStatsRecordingAgent(RL.Agent):
    def __init__(self, context: RL.Context, name):
        super().__init__(context, name)

    def start(self):
        RL.stats.record_start_time()
        self.episode_rewards = np.zeros(self.context.num_envs)
        self.episode_lengths = np.zeros(self.context.num_envs, dtype=np.int)

    def post_act(self):
        self.episode_rewards = self.episode_rewards + self.runner.rewards
        self.episode_lengths = self.episode_lengths + 1

    def post_episode(self, env_id_nos):
        for env_id_no in env_id_nos:
            RL.stats.record_append('Env-{0} Ep Id'.format(env_id_no), self.runner.episode_ids[env_id_no])
            RL.stats.record_append('Env-{0} Ep R'.format(env_id_no), self.episode_rewards[env_id_no])
            RL.stats.record_append('Env-{0} Ep Exploit Mode'.format(env_id_no), self.runner.exploit_modes[env_id_no])
            RL.stats.record_append('Env-{0} Ep L'.format(env_id_no), self.episode_lengths[env_id_no])
            RL.stats.record_append('Env-{0} Total Steps'.format(env_id_no), self.runner.step_ids[env_id_no] + 1)
            RL.stats.record_append('Env-{0} Total Episodes'.format(env_id_no), self.runner.episode_ids[env_id_no] + 1)
            RL.stats.record_time_append('Env-{0} Ep Timestamp'.format(env_id_no))
            rewards = RL.stats.get('Env-{0} Ep R'.format(env_id_no))
            exploit_modes = RL.stats.get('Env-{0} Ep Exploit Mode'.format(env_id_no))
            av_rpe = np.mean(rewards[-100:])
            av_rpe_exploit = np.mean([r for r, m in filter(lambda t: t[1], zip(rewards, exploit_modes))][-100:])
            RL.stats.record('Env-{0} Av Ep R (100)'.format(env_id_no), av_rpe)
            RL.stats.record('Env-{0} Av Exploit Ep R (100)'.format(env_id_no), av_rpe_exploit)

            self.episode_rewards[env_id_no] = 0
            self.episode_lengths[env_id_no] = 0
