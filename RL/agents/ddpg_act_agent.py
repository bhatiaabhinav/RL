import RL
from RL.models.ddpg_model import DDPGModel
import numpy as np


class DDPGActAgent(RL.Agent):
    def __init__(self, context: RL.Context, name):
        super().__init__(context, name)
        self.model = DDPGModel(context, "{0}/model".format(name))

    def exploit_policy(self, model: DDPGModel, states):
        return model.actions(states)

    def act(self):
        exploit_env_ids = list(filter(lambda env_id_no: self.runner.exploit_modes[env_id_no], range(self.context.num_envs)))
        if len(exploit_env_ids) > 0:
            exploit_actions = self.exploit_policy(self.model, list(np.asarray(self.runner.obss)[exploit_env_ids]))
            actions = np.asarray([None] * self.context.num_envs)
            for env_id_no in exploit_env_ids:
                actions[env_id_no] = exploit_actions[env_id_no]
            return actions
