import RL
from RL.common.utils import set_global_seeds


class SeedingAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, seed):
        super().__init__(context, name)
        self.seed = seed
        set_global_seeds(self.seed)

    def start(self):
        for i, env in enumerate(self.context.envs):
            self.context.env.seed(self.seed + i)
