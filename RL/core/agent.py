from .context import Context


class Agent:
    def __init__(self, context: Context, name):
        self.context = context  # type: Context
        self.name = name
        from .runner import Runner
        self.runner = None  # type: Runner
        self.enabled = True

    def start(self):
        pass

    def pre_episode(self, env_id_nos):
        pass

    def pre_act(self):
        pass

    def act(self):
        return None

    def post_act(self):
        pass

    def post_episode(self, env_id_nos):
        pass

    def close(self):
        pass
