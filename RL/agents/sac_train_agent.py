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
        Q = np.min(np.array([self.target_model.Q(i, states, target_actions) for i in range(self.context.num_critics)]), axis=0)
        assert len(Q) == len(states)
        assert len(target_logpis) == len(states)
        V = Q - self.context.alpha * target_logpis
        return V

    def train(self):
        c = self.context
        states, actions, rewards, dones, infos, next_states = self.experience_buffer_agent.experience_buffer.random_experiences_unzipped(c.minibatch_size)
        next_states_V = self.get_target_network_V(next_states)
        desired_Q_values = rewards + (1 - dones.astype(np.int)) * (c.gamma ** c.nsteps) * next_states_V
        if c.clip_td_error:
            Q = self.sac_act_agent.model.Q(states, actions)
            td_errors = desired_Q_values - Q
            td_errors = np.clip(-1, 1)
            desired_Q_values = Q + td_errors
        critic_loss = np.mean([self.sac_act_agent.model.train_critic(i, states, actions, desired_Q_values) for i in range(self.context.num_critics)])
        actor_loss, actor_critic_Q, actor_logstds, actor_logpis = self.sac_act_agent.model.train_actor(states, self.sac_act_agent.model.sample_actions_noise(len(states)), self.context.alpha)
        return critic_loss, actor_loss, np.mean(actor_critic_Q), np.mean(actor_logstds), np.mean(actor_logpis)

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
