import RL
from RL.common.utils import ImagePygletWingow


class BasePygletRenderingAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, episode_interval=None):
        super().__init__(context, name)
        self.window = None  # type: ImagePygletWingow
        self.episode_interval = episode_interval
        if self.episode_interval is None:
            self.episode_interval = self.context.render_interval

    def render(self):
        '''Assumes that self.window is initialized'''
        pass

    def create_pyglet_window(self):
        if self.window is None:
            try:
                self.window = ImagePygletWingow(caption=self.context.env_id + ":" + self.context.experiment_name + ":" + self.name, vsync=self.context.render_vsync)
            except Exception as e:
                RL.logger.error("{0}: Could not create window. Reason = {1}".format(self.name, str(e)))

    def start(self):
        pass

    def pre_episode(self, env_id_nos):
        if self.runner.episode_id % self.episode_interval == 0:
            self.create_pyglet_window()
            self.render()
        else:
            self.close()

    def post_act(self):
        if self.runner.episode_id % self.episode_interval == 0:
            self.create_pyglet_window()
            self.render()
        else:
            self.close()

    def close(self):
        if self.window:
            self.window.close()
            self.window = None
