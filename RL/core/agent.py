import numpy as np
from .context import Context

zero_id = np.array([0])


class Agent:
    def __init__(self, context: Context, name, supports_multiple_envs=True):
        self.context = context  # type: Context
        self.name = name
        from .runner import Runner
        self.runner = None  # type: Runner
        self.enabled = True
        self.supports_multiple_envs = supports_multiple_envs

    def start(self):
        pass

    def pre_episode(self):
        if self.supports_multiple_envs:
            self.pre_episodes(zero_id)

    def pre_episodes(self, env_id_nos):
        if not self.supports_multiple_envs:
            raise NotImplementedError("This agent does not support multiple parallel envs yet")

    def pre_act(self):
        pass

    def act(self):
        if self.supports_multiple_envs:
            acts = self.acts()
            return None if acts is None else acts[0]
        else:
            return None

    def acts(self):
        if not self.supports_multiple_envs:
            raise NotImplementedError("This agent does not support multiple parallel envs yet")

    def post_act(self):
        pass

    def post_episode(self):
        if self.supports_multiple_envs:
            self.post_episodes(zero_id)

    def post_episodes(self, env_id_nos):
        if not self.supports_multiple_envs:
            raise NotImplementedError("This agent does not support multiple parallel envs yet")

    def close(self):
        pass

    def post_close(self):
        pass
