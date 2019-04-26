import RL


class LinearAnnealingAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, variable_name, setter, start_delay, start_val, final_val, duration):
        super().__init__(context, name)
        self.setter = setter
        self.start_delay = start_delay
        self.start_val = start_val
        self.final_val = final_val
        self.duration = duration
        self.variable_name = variable_name

    def start(self):
        self.setter(self.start_val)

    def pre_act(self):
        if self.runner.num_steps <= self.start_delay:
            val = self.start_val
        else:
            steps = self.runner.num_steps - self.start_delay
            val = self.final_val + (self.start_val - self.final_val) * (1 - min(steps / self.duration, 1))
        self.setter(val)
        if self.runner.step_id % 10 == 0:
            RL.stats.record(self.variable_name, val)
