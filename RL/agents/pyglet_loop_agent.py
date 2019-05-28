import RL
import pyglet


class PygletLoopAgent(RL.Agent):
    def __init__(self, context: RL.Context, name):
        super().__init__(context, name)
        if self.context.pyglet_fps > 0:
            pyglet.clock.set_fps_limit(self.context.pyglet_fps)

    def tick(self):
        pyglet.clock.tick()
        if not self.context.auto_dispatch_on_render:
            for window in pyglet.app.windows:
                window.switch_to()
                window.dispatch_events()
                window.dispatch_event('on_draw')
                window.flip()

    def pre_episode(self, env_id_nos):
        self.tick()

    def post_act(self):
        self.tick()
