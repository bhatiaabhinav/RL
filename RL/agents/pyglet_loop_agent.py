import RL
import pyglet


class PygletLoopAgent(RL.Agent):
    """Forces an FPS. Additionally, if context.auto_dispatch_on_render is False, then issues draw & dispatch event command to each window which has needs_draw attribute. Informs the window that draw was called by setting its `needs_draw` flag to false"""
    def __init__(self, context: RL.Context, name):
        super().__init__(context, name)
        if self.context.pyglet_fps > 0:
            pyglet.clock.set_fps_limit(self.context.pyglet_fps)

    def tick(self):
        pyglet.clock.tick()
        if not self.context.auto_dispatch_on_render:
            for window in pyglet.app.windows:
                if hasattr(window, 'needs_draw'):
                    window.switch_to()
                    # window.clear()
                    window.dispatch_events()
                    if getattr(window, 'needs_draw'):
                        setattr(window, 'needs_draw', False)
                        window.dispatch_event('on_draw')
                        window.flip()

    def pre_episodes(self, env_id_nos):
        self.tick()

    def post_act(self):
        self.tick()
