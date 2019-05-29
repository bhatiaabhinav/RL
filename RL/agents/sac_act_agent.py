import RL
from RL.models.sac_model import SACModel
import numpy as np


class SACActAgent(RL.Agent):
    def __init__(self, context: RL.Context, name):
        super().__init__(context, name)
        self.model = SACModel(context, "{0}/model".format(name), num_actors=1, num_critics=self.context.num_critics, num_valuefns=1)

    def policy(self, model: SACModel, states, exploit_modes):
        assert len(exploit_modes) == len(states)
        sigma = 1 - np.asarray(exploit_modes).astype(np.int)
        noise = model.sample_actions_noise(len(states), sigma=sigma)
        return model.actions(states, noise)

    def act(self):
        if not self.context.eval_mode and self.runner.num_steps < self.context.minimum_experience:
            return None  # let the random player act
        else:
            return self.policy(self.model, self.runner.obss, self.context.force_exploits)
