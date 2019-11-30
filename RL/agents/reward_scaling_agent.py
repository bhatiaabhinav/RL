import RL


class RewardScalingAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, reward_scaling=None, cost_scaling=None):
        super().__init__(context, name)
        self.reward_scaling = context.reward_scaling if reward_scaling is None else reward_scaling
        self.cost_scaling = context.cost_scaling if cost_scaling is None else cost_scaling

    def post_act(self):
        self.runner.rewards *= self.reward_scaling
        self.runner.costs *= self.cost_scaling

