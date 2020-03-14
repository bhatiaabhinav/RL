import RL
from RL.models.safe_sac_model import SafeSACModel
import numpy as np


class SafeSACActAgent(RL.Agent):
    def __init__(self, context: RL.Context, name):
        super().__init__(context, name)
        self.model = SafeSACModel(context, "{0}/model".format(name), num_actors=1, num_critics=self.context.num_critics + 2, num_valuefns=2)

    def policy(self, model: SafeSACModel, states, exploit_modes):
        assert len(exploit_modes) == len(states)
        sigma = 1 - np.asarray(exploit_modes).astype(np.int)
        noise = model.sample_actions_noise(len(states), sigma=sigma)
        return model.actions(states, noise)

    def acts(self):
        if not self.context.eval_mode and self.runner.num_steps < self.context.minimum_experience:
            return None  # let the random player act
        else:
            return self.policy(self.model, self.runner.obss, self.context.force_exploits)
