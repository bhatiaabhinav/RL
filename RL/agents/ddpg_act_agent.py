import RL
from RL.models.ddpg_model import DDPGModel


class DDPGActAgent(RL.Agent):
    def __init__(self, context: RL.Context, name):
        super().__init__(context, name)
        self.model = DDPGModel(context, "{0}/model".format(name))

    def exploit_policy(self, model: DDPGModel, states):
        return model.actions(states)

    def act(self):
        if self.context.should_eval_episode():
            return self.exploit_policy(self.model, [self.context.frame_obs])[0]
