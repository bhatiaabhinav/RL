import RL
from .minmax_agent import MinMaxAgent
from RL.models.value_fn_approximator import ValueFunctionApproximator
import tensorflow as tf
import numpy as np
from RL.common.experience_buffer import Experience, ExperienceBuffer


class MinMaxNetsAgent(MinMaxAgent):
    def __init__(self, context: RL.Context, name, model_name, max_depth=5, model_reuse=tf.AUTO_REUSE, prune=True):
        super().__init__(context, name, prune=prune)
        input_shape = [self.board_size * self.board_size * 3]
        input_dtype = np.uint8
        self.max_depth = max_depth
        self.model_name = model_name
        self.model = ValueFunctionApproximator(context, model_name, input_shape, input_dtype, need_conv=False, reuse=model_reuse)
        self.model.setup_training("traning_" + self.model.name, reuse=model_reuse)
        self.replay_memory = ExperienceBuffer(length=context.experience_buffer_length)

    def Vs(self, turns, boards, depth=0, prune_fn=None):
        boards_flat = [b.flatten() for b in boards]
        if depth > self.max_depth:
            Vs_p0 = np.clip(self.model.get_V(boards_flat), -1, 1)
            Vs_p1 = -Vs_p0
            Vs = np.asarray(list(zip(Vs_p0, Vs_p1)))
        else:
            Vs = super().Vs(turns, boards, depth=depth, prune_fn=prune_fn)
            for b, V in zip(boards_flat, list(Vs)):
                self.replay_memory.add(Experience(b, None, V[0], False, None, None))
        return Vs

    def post_act(self):
        count = min(self.context.minibatch_size, self.replay_memory.count)
        boards, a, Vs, d, i, s1 = self.replay_memory.random_experiences_unzipped(count)
        loss = self.model.train_V(boards, Vs)
        if self.context.frame_id == 0:
            self.context.summaries.setup_scalar_summaries(['loss'])
        if self.context.frame_id % 10 == 0:
            self.context.log_summary({"loss": loss}, self.context.frame_id)
