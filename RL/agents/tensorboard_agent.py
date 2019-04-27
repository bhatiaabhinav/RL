import RL
import numpy as np


class TensorboardAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, keys, x_key, keys_prefix='', log_every_episode=1, log_every_step=1e12):
        super().__init__(context, name)
        self.keys = keys
        self.x_key = x_key
        self.keys_prefix = keys_prefix
        self.log_every_episode = log_every_episode
        self.log_every_step = log_every_step
        self._need_setup = True

    def start(self):
        self.context.summaries.setup_scalar_summaries(self.keys)

    def post_act(self):
        if np.any(self.runner.step_ids % self.log_every_step == 0):
            self.read_and_push_to_tb()

    def post_episode(self, env_id_nos):
        if np.any(self.runner.episode_ids[env_id_nos] % self.log_every_episode == 0):
            self.read_and_push_to_tb()

    def close(self):
        self.read_and_push_to_tb()

    def read_and_push_to_tb(self):
        x_key = self.keys_prefix + self.x_key
        x = RL.stats.get(x_key)
        if RL.stats.get_key_type(x_key) == RL.Stats.KeyType.LIST:
            x = x[-1]
        kvs = {}
        for key in self.keys:
            full_key = self.keys_prefix + key
            val = RL.stats.get(full_key)
            if RL.stats.get_key_type(full_key) == RL.Stats.KeyType.LIST:
                val = val[-1]
            kvs[key] = val
        self.context.summaries.write_summaries(kvs, x)
