import RL
from RL.agents.sac_act_agent import SACActAgent
from RL.agents.experience_buffer_agent import ExperienceBufferAgent
from RL.models.sac_model import SACModel
import numpy as np


class SACTrainAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, sac_act_agent: SACActAgent, exp_buffer_agent: ExperienceBufferAgent):
        super().__init__(context, name)
        self.sac_act_agent = sac_act_agent
        self.experience_buffer_agent = exp_buffer_agent
        self.target_model = SACModel(context, "{0}/target_model".format(name), num_critics=self.context.num_critics)
        self.sac_act_agent.model.setup_training("{0}/training_ops".format(name))
        self.total_updates = 0

    def pre_act(self):
        if self.context.normalize_observations:
            self.sac_act_agent.model.update_states_running_stats(self.runner.obss)

    def get_target_network_V(self, states):
        target_actions, _, _, target_logpis = self.target_model.actions_means_logstds_logpis(states, noise=self.target_model.sample_actions_noise(len(states)))
        Q = np.min(np.asarray(self.target_model.Q(list(range(self.context.num_critics)), states, target_actions)), axis=0)
        assert len(Q) == len(states)
        assert len(target_logpis) == len(states)
        V = Q - self.context.alpha * target_logpis
        return V

    def train(self):
        c = self.context
        states, actions, rewards, dones, infos, next_states = self.experience_buffer_agent.experience_buffer.random_experiences_unzipped(c.minibatch_size)
        next_states_V = self.get_target_network_V(next_states)
        desired_Q_values = rewards + (1 - dones.astype(np.int)) * (c.gamma ** c.nsteps) * next_states_V
        desired_Q_values_per_critic = [desired_Q_values] * c.num_critics
        # TODO: implement TD error clipping
        critic_loss = self.sac_act_agent.model.train_critics(states, actions, list(range(c.num_critics)), desired_Q_values_per_critic, [1 / c.num_critics] * c.num_critics)
        noise = self.sac_act_agent.model.sample_actions_noise(len(states))
        actor_loss, actor_critics_Qs, actor_logstds, actor_logpis = self.sac_act_agent.model.train_actor(states, noise, [0], [1], c.alpha)
        return critic_loss, actor_loss, np.mean(actor_critics_Qs[0]), np.mean(actor_logstds), np.mean(actor_logpis)

    def post_act(self):
        c = self.context
        if self.context.normalize_actions:
            self.sac_act_agent.model.update_actions_running_stats(self.runner.actions)
        if self.runner.step_id % c.train_every == 0 and self.runner.step_id >= c.minimum_experience:
            critic_loss, actor_loss, av_actor_critic_Q, av_logstd, av_logpi = 0, 0, 0, 0, 0
            for sgd_step_id in range(self.context.gradient_steps):
                critic_loss, actor_loss, av_actor_critic_Q, av_logstd, av_logpi = self.train()
            self.total_updates += self.context.gradient_steps
            if self.runner.step_id % 10 == 0:
                RL.stats.record_append("Total Updates", self.total_updates)
                RL.stats.record_append("Critic Loss", critic_loss)
                RL.stats.record_append("Actor Loss", actor_loss)
                RL.stats.record_append("Average Actor Critic Q", av_actor_critic_Q)
                RL.stats.record_append("Average Action LogStd", av_logstd)
                RL.stats.record_append("Average Action LogPi", av_logpi)
