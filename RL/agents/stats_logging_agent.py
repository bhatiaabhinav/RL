import RL
import numpy as np


class StatsLoggingAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, keys, poll_every_episode=1, poll_every_step=1e9, keys_prefix=''):
        super().__init__(context, name)
        self.keys = keys
        self.keys_prefix = keys_prefix
        self.key_vals = {}
        self._changes = False
        self.poll_every_episode = poll_every_episode
        self.poll_every_step = poll_every_step

    def post_act(self):
        if np.any(self.runner.step_ids % self.poll_every_step == 0):
            self.read_key_vals()
            if self._changes:
                self.write_key_vals()

    def post_episode(self, env_id_nos):
        if np.any(self.runner.episode_ids[env_id_nos] % self.poll_every_episode == 0):
            self.read_key_vals()
            if self._changes:
                self.write_key_vals()

    def close(self):
        self.read_key_vals()
        self.write_key_vals()

    def read_key_vals(self):
        self._changes = False
        for key in self.keys:
            full_key = self.keys_prefix + key
            val = RL.stats.get(full_key)
            if RL.stats.get_key_type(full_key) == RL.Stats.KeyType.LIST:
                val = val[-1]
            old_val = self.key_vals.get(key)
            if not val == old_val:
                self._changes = True
            self.key_vals[key] = val

    def write_key_vals(self):
        RL.logger.logkvs(self.key_vals)
        RL.logger.dumpkvs()
