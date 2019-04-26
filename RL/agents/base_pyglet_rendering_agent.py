import RL
from RL.common.utils import ImagePygletWingow


class BasePygletRenderingAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, auto_dispatch_on_render=True, episode_interval=1):
        super().__init__(context, name)
        self.vsync = context.render_vsync
        self.window = None
        self.auto_dispatch_on_render = auto_dispatch_on_render
        self.episode_interval = episode_interval

    def render(self):
        pass

    def start(self):
        try:
            self.window = ImagePygletWingow(caption=self.context.env_id + ":" + self.context.experiment_name + ":" + self.name, vsync=self.vsync)
        except Exception as e:
            RL.logger.error("{0}: Could not create window. Reason = {1}".format(self.name, str(e)))

    def pre_episode(self, env_id_nos):
        if self.window and self.runner.episode_id % self.episode_interval == 0:
            self.render()

    def post_act(self):
        if self.window and self.runner.episode_id % self.episode_interval == 0:
            self.render()

    def close(self):
        if self.window:
            self.window.close()
