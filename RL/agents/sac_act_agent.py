import RL
from RL.models.sac_model import SACModel
import numpy as np


class SACActAgent(RL.Agent):
    def __init__(self, context: RL.Context, name):
        super().__init__(context, name)
        self.model = SACModel(context, "{0}/model".format(name), num_critics=self.context.num_critics)

    def policy(self, model: SACModel, states, exploit_modes):
        assert len(exploit_modes) == len(states)
        sigma = 1 - np.asarray(exploit_modes).astype(np.int)
        noise = model.sample_actions_noise(len(states), sigma=sigma)
        return model.actions(states, noise)

    def act(self):
        if self.runner.num_steps < self.context.minimum_experience:
            return [self.context.envs[i].action_space.sample() for i in range(self.context.num_envs)]
        else:
            return self.policy(self.model, self.runner.obss, self.runner.exploit_modes)
