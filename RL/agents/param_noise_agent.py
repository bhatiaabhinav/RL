import RL
from RL.common.utils import TFParamsCopier
import numpy as np


class ParamNoiseAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, target_model, source_model, target_divergence=None, adaptation_factor=None):
        super().__init__(context, name)
        self.sigma = 0.001
        self.target_divergence = target_divergence
        self.adaptation_factor = adaptation_factor
        self.source_model = source_model
        self.model = target_model
        self.perturb_copier = TFParamsCopier("{0}/perturbable_params_copier".format(self.name), self.source_model.get_perturbable_vars(), self.model.get_perturbable_vars(), self.context.session)
        self.all_copier = TFParamsCopier("{0}/all_params_copier".format(self.name), self.source_model.params, self.model.params, self.context.session)
        self.episode_states = []
        if self.target_divergence is None:
            self.target_divergence = self.context.param_noise_divergence
        if self.adaptation_factor is None:
            self.adaptation_factor = self.context.param_noise_adaptation_factor

    def act(self):
        if self.runner.num_steps >= self.context.minimum_experience:
            explore_env_ids = list(filter(lambda env_id_no: not self.runner.exploit_modes[env_id_no], range(self.context.num_envs)))
            if len(explore_env_ids) > 0:
                explore_actions = self.model.actions(list(np.asarray(self.runner.obss)[explore_env_ids]))
                actions = np.asarray([None] * self.context.num_envs)
                for env_id_no in explore_env_ids:
                    actions[env_id_no] = explore_actions[env_id_no]
                self.episode_states.extend(self.runner.obss)
                return actions

    def copy(self):
        self.all_copier.copy(tau=1)
        self.perturb_copier.copy(tau=1, noise=self.perturb_copier.generate_normal_noise(self.sigma))

    def pre_episode(self, env_id_nos):
        if self.runner.num_steps >= self.context.minimum_experience and 0 in env_id_nos and not self.runner.exploit_modes[0]:
            self.copy()
            self.episode_states.clear()

    def post_episode(self, env_id_nos):
        if self.runner.num_steps >= self.context.minimum_experience and 0 in env_id_nos and not self.runner.exploit_modes[0]:
            sampled_state_ids = np.random.choice(len(self.episode_states), self.context.minibatch_size)
            sampled_states = np.asarray(self.episode_states)[sampled_state_ids]
            my_actions = self.model.actions(sampled_states)
            source_actions = self.source_model.actions(sampled_states)
            divergence = np.sqrt(np.mean(np.square(my_actions - source_actions)))
            if divergence > self.target_divergence:
                self.sigma = self.sigma / self.adaptation_factor
            else:
                self.sigma = self.adaptation_factor * self.sigma
            RL.stats.record_append("Exploration Divergence", divergence)
            RL.stats.record_append("Exploration Sigma", self.sigma)
