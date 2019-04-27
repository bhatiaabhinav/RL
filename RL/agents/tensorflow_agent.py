import RL
import tensorflow as tf
from RL.common.summaries import Summaries


class TensorFlowAgent(RL.Agent):
    def __init__(self, context: RL.Context, name):
        super().__init__(context, name)
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = tf.Session(config=self.config)
        self.context.session = self.session

    def start(self):
        super().start()
        self.summaries = Summaries(self.session)
        self.context.summaries = self.summaries
        self.session.run(tf.global_variables_initializer())

    def post_close(self):
        super().post_close()
        self.session.close()
