import RL
import numpy as np


class StatsLoggingAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, keys, poll_every_episode=1, poll_every_step=-1, keys_prefix=''):
        super().__init__(context, name)
        self.keys = keys
        self.friendly_keys = [k for k in self.keys]
        self.keys_prefix = keys_prefix
        for i, (k, fk) in enumerate(zip(self.keys, self.friendly_keys)):
            if ':' in k:
                fk = k.split(':')[0]
                k = ''.join(k.split(':')[1:])
            else:
                fk = k
            k = keys_prefix + k
            self.keys[i] = k
            self.friendly_keys[i] = fk
        self.friendly_key_vals = {}
        self._changes = False
        self.poll_every_episode = poll_every_episode
        self.poll_every_step = poll_every_step

    def post_act(self):
        if self.poll_every_step > 0 and np.any(self.runner.step_ids % self.poll_every_step == 0):
            self.read_key_vals()
            if self._changes:
                self.write_key_vals()

    def post_episode(self, env_id_nos):
        if self.poll_every_episode > 0 and np.any(self.runner.episode_ids[env_id_nos] % self.poll_every_episode == 0):
            self.read_key_vals()
            if self._changes:
                self.write_key_vals()

    def close(self):
        self.read_key_vals()
        self.write_key_vals()

    def read_key_vals(self):
        self._changes = False
        for key, friendly_key in zip(self.keys, self.friendly_keys):
            val = RL.stats.get(key)
            if RL.stats.get_key_type(key) == RL.Stats.KeyType.LIST:
                if len(val) > 0:
                    val = val[-1]
                else:
                    val = None
            old_val = self.friendly_key_vals.get(friendly_key)
            if not val == old_val:
                self._changes = True
            self.friendly_key_vals[friendly_key] = val

    def write_key_vals(self):
        RL.logger.logkvs(self.friendly_key_vals)
        RL.logger.dumpkvs()
