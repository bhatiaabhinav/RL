import RL
from RL.common.utils import TFParamsCopier


class ParamsCopyAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, params_from, params_to, interval, tau):
        super().__init__(context, name)
        self.interval = interval
        self.tau = tau
        self.copier = TFParamsCopier(name, params_from, params_to, self.context.session)

    def start(self):
        super().start()
        self.copier.copy()

    def post_act(self):
        if self.runner.num_steps >= self.context.minimum_experience and self.runner.step_id % self.interval == 0:
            self.copier.copy(tau=self.tau)
