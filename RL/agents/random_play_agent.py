import RL


class RandomPlayAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, play_for_steps=None):
        super().__init__(context, name)
        self.play_for_steps = play_for_steps

    def acts(self):
        if self.play_for_steps is not None and self.runner.num_steps >= self.play_for_steps:
            return None
        else:
            return [self.context.env.action_space.sample() for env in self.context.envs]
