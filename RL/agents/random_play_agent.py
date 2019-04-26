import RL


class RandomPlayAgent(RL.Agent):
    def act(self):
        return [self.context.env.action_space.sample() for env in self.context.envs]
