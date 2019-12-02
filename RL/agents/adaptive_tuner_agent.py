import RL
import numpy as np


class AdaptiveParamTunerAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, variable_name, start_delay, start_val, min_value, max_value, lr, signal_fn, adaptation_style='additive'):
        super().__init__(context, name)
        self.start_delay = start_delay
        self.start_val = start_val
        self.min_value = min_value
        self.max_value = max_value
        self.lr = lr
        if adaptation_style == 'additive':
            self.adapt_fn = self.additive_adapt_fn
        elif adaptation_style == 'multiplicative':
            self.adapt_fn = self.multiplicative_adapt_fn
        self.signal_fn = signal_fn
        self.variable_name = variable_name

    def additive_adapt_fn(self, x, incr, signal):
        x += np.sign(signal) * np.minimum(np.maximum(np.abs(signal) * incr, incr), 10 * incr)
        return x

    def multiplicative_adapt_fn(self, x, multiplier, signal):
        if signal > 0:
            x = x * multiplier
        elif signal < 0:
            x = x / multiplier
        else:
            pass
        return x

    def start(self):
        self.context.set_attribute(self.variable_name, self.start_val)

    def post_episode(self):
        if self.runner.num_steps <= self.start_delay:
            val = self.start_val
        else:
            val = self.context.get_attribute(self.variable_name)
            val = self.adapt_fn(val, self.lr, self.signal_fn())
            val = np.maximum(np.minimum(val, self.max_value), self.min_value)
        self.context.set_attribute(self.variable_name, val)
        RL.stats.record(self.variable_name, val)
