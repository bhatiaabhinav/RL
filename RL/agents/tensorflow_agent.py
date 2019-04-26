import RL
import tensorflow as tf


class TensorFlowAgent(RL.Agent):
    def __init__(self, context: RL.Context, name):
        super().__init__(context, name)
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = tf.Session(config=self.config)
        self.context.session = self.session

    def start(self):
        super().start()
        self.session.run(tf.global_variables_initializer())

    def close(self):
        super().close()
        self.session.close()
