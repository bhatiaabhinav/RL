import RL


class RewardScalingAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, reward_scaling=None):
        super().__init__(context, name)
        self.reward_scaling = context.reward_scaling if reward_scaling is None else reward_scaling

    def post_act(self):
        self.runner.rewards *= self.reward_scaling
        for info in self.runner.infos:
            info['Safety_reward'] = info.get('Safety_reward', 0) * self.reward_scaling
