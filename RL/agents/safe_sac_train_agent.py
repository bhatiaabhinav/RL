import RL
from RL.agents.sac_act_agent import SACActAgent
from RL.agents.experience_buffer_agent import ExperienceBufferAgent
from RL.models.safe_sac_model import SafeSACModel
import numpy as np


class SafeSACTrainAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, sac_act_agent: SACActAgent, exp_buffer_agent: ExperienceBufferAgent):
        super().__init__(context, name)
        self.sac_act_agent = sac_act_agent
        self.experience_buffer_agent = exp_buffer_agent
        self.experience_buffer_agent.get_reward = self.get_reward
        assert len(self.context.safety_stream_names) == 1, "Right now exactly one safety stream is supported"
        self.target_model = SafeSACModel(context, "{0}/target_model".format(name), num_actors=0, num_critics=0, num_valuefns=2)
        self.sac_act_agent.model.setup_training("{0}/training_ops".format(name))
        self.total_updates = 0
        self.critic_ids = list(range(self.context.num_critics + 1))

    def get_reward(self, env_id_no):
        return [self.runner.rewards[env_id_no]] + [self.runner.infos[env_id_no][stream_name + '_reward'] for stream_name in self.context.safety_stream_names]

    def pre_act(self):
        if self.context.normalize_observations:
            self.sac_act_agent.model.update_states_running_stats(self.runner.obss)

    def get_V_and_safety_V_targets(self, states, noise):
        actions, _, _, logpis = self.sac_act_agent.model.actions_means_logstds_logpis(states, noise)
        all_Qs = self.sac_act_agent.model.Q(self.critic_ids, states, actions)
        Q = np.min(np.asarray(all_Qs[:self.context.num_critics]), axis=0)
        V_targets = Q - self.context.alpha * logpis
        safety_V_targets = all_Qs[-1]
        assert len(Q) == len(states)
        return V_targets, safety_V_targets

    def get_Q_targets_and_safety_Q_targets(self, multistream_rewards, dones, next_states):
        c = self.context
        rewards = multistream_rewards[:, 0]
        safety_rewards = multistream_rewards[:, 1]
        next_states_all_V = self.target_model.V([0, 1], next_states)
        next_states_V = next_states_all_V[0]
        next_states_safety_V = next_states_all_V[1]
        Q_targets = rewards + (1 - dones.astype(np.int)) * (c.gamma ** c.nsteps) * next_states_V
        safety_Q_targets = safety_rewards + (1 - dones.astype(np.int)) * (c.safety_gamma ** c.nsteps) * next_states_safety_V
        return Q_targets, safety_Q_targets

    def train(self):
        c = self.context
        states, actions, multistream_rewards, dones, infos, next_states = self.experience_buffer_agent.experience_buffer.random_experiences_unzipped(c.minibatch_size)
        noise = self.sac_act_agent.model.sample_actions_noise(len(states))
        V_targets, safety_V_targets = self.get_V_and_safety_V_targets(states, noise)
        Q_targets, safety_Q_targets = self.get_Q_targets_and_safety_Q_targets(multistream_rewards, dones, next_states)
        Q_targets_per_critic = [Q_targets] * c.num_critics
        # TODO: implement TD error clipping
        valuefn_loss = self.sac_act_agent.model.train_valuefns(states, [0], [V_targets], [1])
        safety_valuefn_loss = self.sac_act_agent.model.train_valuefns(states, [1], [safety_V_targets], [1])
        critic_loss = self.sac_act_agent.model.train_critics(states, actions, self.critic_ids[:c.num_critics], Q_targets_per_critic, [1 / c.num_critics] * c.num_critics)
        safety_critic_loss = self.sac_act_agent.model.train_critics(states, actions, [c.num_critics], [safety_Q_targets], [1])
        actor_loss, actor_critics_Qs, actor_logstds, actor_logpis = self.sac_act_agent.model.train_actor(states, noise, [0, c.num_critics], [1, 1], c.alpha)
        return valuefn_loss, safety_valuefn_loss, critic_loss, safety_critic_loss, actor_loss, np.mean(actor_critics_Qs[0]), np.mean(actor_critics_Qs[1]), np.mean(actor_logstds), np.mean(actor_logpis)

    def post_act(self):
        c = self.context
        if self.context.normalize_actions:
            self.sac_act_agent.model.update_actions_running_stats(self.runner.actions)
        if self.runner.step_id % c.train_every == 0 and self.runner.step_id >= c.minimum_experience:
            valuefn_loss, safety_valuefn_loss, critic_loss, safety_critic_loss, actor_loss, av_actor_critic_Q, av_actor_critic_safety_Q, av_logstd, av_logpi = 0, 0, 0, 0, 0, 0, 0, 0, 0
            for sgd_step_id in range(self.context.gradient_steps):
                valuefn_loss, safety_valuefn_loss, critic_loss, safety_critic_loss, actor_loss, av_actor_critic_Q, av_actor_critic_safety_Q, av_logstd, av_logpi = self.train()
            self.total_updates += self.context.gradient_steps
            if self.runner.step_id % 10 == 0:
                RL.stats.record("Total Updates", self.total_updates)
                RL.stats.record("ValueFn Loss", valuefn_loss)
                RL.stats.record("Safety ValueFn Loss", safety_valuefn_loss)
                RL.stats.record("Critic Loss", critic_loss)
                RL.stats.record("Safety Critic Loss", safety_critic_loss)
                RL.stats.record("Actor Loss", actor_loss)
                RL.stats.record("Average Actor Critic Q", av_actor_critic_Q)
                RL.stats.record("Average Actor Critic Safety Q", av_actor_critic_safety_Q)
                RL.stats.record("Average Action LogStd", av_logstd)
                RL.stats.record("Average Action LogPi", av_logpi)
