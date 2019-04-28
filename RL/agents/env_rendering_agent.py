import RL
from .base_pyglet_rendering_agent import BasePygletRenderingAgent


class EnvRenderingAgent(BasePygletRenderingAgent):
    def __init__(self, context: RL.Context, name, auto_dispatch_on_render=True, episode_interval=None):
        super().__init__(context, name, auto_dispatch_on_render=auto_dispatch_on_render, episode_interval=episode_interval)
        self.text_mode = 'ansi' in self.context.env.metadata.get('render.modes', [])

    def render(self):
        if self.context.render_mode == 'auto':
            if not self.text_mode:
                self.window.set_image(self.context.env.render(mode='rgb_array'), self.auto_dispatch_on_render)
            else:
                self.window.set_text_image(self.context.env.render(mode='ansi'), self.auto_dispatch_on_render)
        else:
            self.context.env.render(mode=self.context.render_mode)

    def start(self):
        if self.context.render_mode == 'auto':
            super().start()
        else:
            self.window = True
