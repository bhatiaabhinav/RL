import RL
import numpy as np
import numbers


class TensorboardAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, keys, x_key, keys_prefix='', log_every_episode=1, log_every_step=-1):
        super().__init__(context, name)
        self.keys = keys
        self.friendly_keys = [k for k in self.keys]
        self.x_key = keys_prefix + x_key
        self.keys_prefix = keys_prefix
        self.log_every_episode = log_every_episode
        self.log_every_step = log_every_step
        self.hist_keys = set()
        for i, (k, fk) in enumerate(zip(self.keys, self.friendly_keys)):
            if ':' in k:
                fk = k.split(':')[0]  # type: str
                k = ''.join(k.split(':')[1:])
            else:
                fk = k  # type: str
            k = keys_prefix + k
            fk = fk.replace(' ', '_')
            if k.endswith('_hist'):
                self.hist_keys.add(k)
            self.keys[i] = k
            self.friendly_keys[i] = fk

    def start(self):
        for k, fk in zip(self.keys, self.friendly_keys):
            if k in self.hist_keys:
                self.context.summaries.setup_histogram_summaries([fk])
            else:
                self.context.summaries.setup_scalar_summaries([fk])

    def post_act(self):
        if self.log_every_step > 0 and np.any(self.runner.step_ids % self.log_every_step == 0):
            self.read_and_push_to_tb()

    def post_episodes(self, env_id_nos):
        if self.log_every_episode > 0 and np.any(self.runner.episode_ids[env_id_nos] % self.log_every_episode == 0):
            self.read_and_push_to_tb()

    def close(self):
        self.read_and_push_to_tb()

    def read_and_push_to_tb(self):
        x = RL.stats.get(self.x_key)
        if x is None:
            return
        if RL.stats.get_key_type(self.x_key) == RL.Stats.KeyType.LIST:
            if len(x) > 0:
                x = x[-1]
            else:
                return
        kvs = {}
        for key, friendly_key in zip(self.keys, self.friendly_keys):
            val = RL.stats.get(key)
            if val is None:
                continue
            if RL.stats.get_key_type(key) == RL.Stats.KeyType.LIST:
                if len(val) > 0:
                    val = val[-1]
                else:
                    continue
            if val is not None:
                if isinstance(val, list) or isinstance(val, np.ndarray):
                    kvs[friendly_key] = np.asarray(val).astype(np.float32)
                elif isinstance(val, numbers.Number):
                    kvs[friendly_key] = float(val)
        self.context.summaries.write_summaries(kvs, x)
