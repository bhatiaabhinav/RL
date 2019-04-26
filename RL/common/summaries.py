import RL
import tensorflow as tf


class Summaries:
    def __init__(self, session: tf.Session):
        RL.logger.log("Setting up summaries")
        self.session = session
        self.writer = tf.summary.FileWriter(
            RL.logger.get_dir(), self.session.graph)

    def setup_scalar_summaries(self, keys):
        for k in keys:
            # ensure no white spaces in k:
            if ' ' in k:
                raise ValueError("Keys cannot contain whitespaces")
            placeholder_symbol = k
            setattr(self, placeholder_symbol, tf.placeholder(
                dtype=tf.float32, name=placeholder_symbol + '_placeholder'))
            placeholder = getattr(self, placeholder_symbol)
            summay_symbol = k + '_summary'
            setattr(self, summay_symbol, tf.summary.scalar(k, placeholder))

    def setup_histogram_summaries(self, keys):
        for k in keys:
            # ensure no white spaces in k:
            if ' ' in k:
                raise ValueError("Keys cannot contain whitespaces")
            placeholder_symbol = k
            setattr(self, placeholder_symbol, tf.placeholder(
                dtype=tf.float32, shape=[None], name=placeholder_symbol + '_placeholder'))
            placeholder = getattr(self, placeholder_symbol)
            summay_symbol = k + '_summary'
            setattr(self, summay_symbol, tf.summary.histogram(k, placeholder))

    def write_summaries(self, kvs, global_step):
        for key in kvs:
            placeholder_symbol = key
            summary_symbol = key + "_summary"
            if hasattr(self, placeholder_symbol) and hasattr(self, summary_symbol):
                summary = self.session.run(getattr(self, summary_symbol), feed_dict={
                    getattr(self, placeholder_symbol): kvs[key]
                })
                self.writer.add_summary(summary, global_step=global_step)
            else:
                RL.logger.log("Invalid summary key {0}".format(
                    key), level=RL.logger.WARN)
