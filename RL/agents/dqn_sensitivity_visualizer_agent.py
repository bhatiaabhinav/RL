import RL
from RL.models.dqn_model import Brain
from .base_pyglet_rendering_agent import BasePygletRenderingAgent
import numpy as np
from PIL import Image
from RL.common.utils import need_conv_net


class DQNSensitivityVisualizerAgent(BasePygletRenderingAgent):
    def __init__(self, context: RL.Context, name, dqn_model: Brain, head_id=0, episode_interval=None):
        super().__init__(context, name, episode_interval)
        self.model = dqn_model  # type: Brain
        self.head_id = head_id

    def render(self):
        context = self.context
        frame = self.model.get_Q_input_sensitivity([self.runner.obs])[self.head_id][0]
        channels = frame.shape[2]
        frame = np.dot(frame.astype('float32'), np.ones([channels]) / channels)
        frame = np.expand_dims(frame, 2)
        frame = np.concatenate([frame] * 3, axis=2)
        frame = (frame * 255).astype(np.uint8)
        orig = context.env.render(mode='rgb_array')
        h, w = orig.shape[0], orig.shape[1]
        frame = np.asarray(Image.fromarray(
            frame).resize((w, h), resample=Image.BILINEAR))
        mixed = 0.9 * frame + 0.1 * orig
        self.window.set_image(mixed.astype(np.uint8), self.context.auto_dispatch_on_render)

    def start(self):
        if need_conv_net(self.context.env.observation_space):
            super().start()
