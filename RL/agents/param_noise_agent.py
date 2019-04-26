import RL
from RL.common.utils import TFParamsCopier
import numpy as np


class ParamNoiseAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, modal_class, source_model, target_divergence=0.2):
        super().__init__(context, name)
        self.sigma = target_divergence
        self.target_divergence = target_divergence
        self.source_model = source_model
        self.model = modal_class(context, "{0}/noisy_model".format(self.name))
        self.perturb_copier = TFParamsCopier("{0}/perturbable_params_copier".format(self.name), self.source_model.get_perturbable_vars(), self.model.get_perturbable_vars(), self.context.session)
        self.all_copier = TFParamsCopier("{0}/all_params_copier".format(self.name), self.source_model.params, self.model.params, self.context.session)
        self.episode_states = []

    def act(self):
        if not self.context.should_eval_episode():
            self.episode_states.append(self.context.frame_obs)
            return self.model.actions([self.context.frame_obs])[0]

    def copy(self):
        self.all_copier.copy(tau=1)
        self.perturb_copier.copy(tau=1, noise=self.perturb_copier.generate_normal_noise(self.sigma))

    def pre_episode(self):
        if not self.context.should_eval_episode():
            self.copy()
            self.episode_states.clear()

    def post_episode(self):
        if not self.context.should_eval_episode():
            my_actions = self.model.actions(self.episode_states)
            source_actions = self.source_model.actions(self.episode_states)
            divergence = np.sqrt(np.mean(np.square(my_actions - source_actions)))
            if divergence > self.target_divergence:
                self.sigma = self.sigma / 1.05
            else:
                self.sigma = 1.05 * self.sigma
