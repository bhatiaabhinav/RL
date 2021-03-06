import RL
from .base_pyglet_rendering_agent import BasePygletRenderingAgent


class EnvRenderingAgent(BasePygletRenderingAgent):
    def __init__(self, context: RL.Context, name, auto_dispatch_on_render=None, episode_interval=None):
        super().__init__(context, name, episode_interval=episode_interval)
        self.auto_dispatch_on_render = self.context.auto_dispatch_on_render if auto_dispatch_on_render is None else auto_dispatch_on_render
        try:
            self.text_mode = 'ansi' in self.context.env.metadata.get('render.modes', [])
        except Exception:
            self.text_mode = False

    def render(self):
        if self.context.render_mode == 'auto':
            if not self.text_mode:
                self.window.set_image(self.context.env.render(mode='rgb_array'), self.auto_dispatch_on_render)
            else:
                self.window.set_text_image(self.context.env.render(mode='ansi'), self.auto_dispatch_on_render)
        else:
            self.context.env.render(mode=self.context.render_mode)

    def create_pyglet_window(self):
        if self.context.render_mode == 'auto':
            super().create_pyglet_window()
        else:
            self.window = self.context.env

    def close(self):
        super().close()
        self.context.env.close()
