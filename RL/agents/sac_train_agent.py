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
        self.target_model = SACModel(context, "{0}/target_model".format(name), num_actors=1, num_critics=self.context.num_critics, num_valuefns=1)
        self.sac_act_agent.model.setup_training("{0}/training_ops".format(name))
        self.total_updates = 0
        self.critic_ids = list(range(self.context.num_critics))

    def pre_act(self):
        if self.context.normalize_observations:
            self.sac_act_agent.model.update_states_running_stats(self.runner.obss)

    def get_V_targets(self, states, noise):
        actions, _, _, logpis = self.sac_act_agent.model.actions_means_logstds_logpis(states, noise)
        Q = np.min(np.asarray(self.sac_act_agent.model.Q(self.critic_ids, states, actions)), axis=0)
        assert len(Q) == len(states)
        return Q - self.context.alpha * logpis

    def get_Q_targets(self, rewards, dones, next_states):
        c = self.context
        next_states_V = self.target_model.V([0], next_states)[0]
        return rewards + (1 - dones.astype(np.int)) * (c.gamma ** c.nsteps) * next_states_V

    def train(self):
        c = self.context
        states, actions, rewards, dones, infos, next_states = self.experience_buffer_agent.experience_buffer.random_experiences_unzipped(c.minibatch_size)
        noise = self.sac_act_agent.model.sample_actions_noise(len(states))
        V_targets = self.get_V_targets(states, noise)
        Q_targets = self.get_Q_targets(rewards, dones, next_states)
        Q_targets_per_critic = [Q_targets] * c.num_critics
        # TODO: implement TD error clipping
        valuefn_loss = self.sac_act_agent.model.train_valuefns(states, [0], [V_targets], [1])
        critic_loss = self.sac_act_agent.model.train_critics(states, actions, self.critic_ids, Q_targets_per_critic, [1 / c.num_critics] * c.num_critics)
        actor_loss, actor_critics_Qs, actor_logstds, actor_logpis = self.sac_act_agent.model.train_actor(states, noise, [0], [1], c.alpha)
        return valuefn_loss, critic_loss, actor_loss, np.mean(actor_critics_Qs[0]), np.mean(actor_logstds), np.mean(actor_logpis)

    def post_act(self):
        c = self.context
        if self.context.normalize_actions:
            self.sac_act_agent.model.update_actions_running_stats(self.runner.actions)
        if self.runner.step_id % c.train_every == 0 and self.runner.step_id >= c.minimum_experience:
            valuefn_loss, critic_loss, actor_loss, av_actor_critic_Q, av_logstd, av_logpi = 0, 0, 0, 0, 0, 0
            for sgd_step_id in range(self.context.gradient_steps):
                valuefn_loss, critic_loss, actor_loss, av_actor_critic_Q, av_logstd, av_logpi = self.train()
            self.total_updates += self.context.gradient_steps
            if self.runner.step_id % 10 == 0:
                RL.stats.record("Total Updates", self.total_updates)
                RL.stats.record("ValueFn Loss", valuefn_loss)
                RL.stats.record("Critic Loss", critic_loss)
                RL.stats.record("Actor Loss", actor_loss)
                RL.stats.record("Average Actor Critic Q", av_actor_critic_Q)
                RL.stats.record("Average Action LogStd", av_logstd)
                RL.stats.record("Average Action LogPi", av_logpi)
