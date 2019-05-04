import RL
from RL.agents.sac_act_agent import SACActAgent
from RL.agents.experience_buffer_agent import ExperienceBufferAgent
from RL.common.experience_buffer import MultiRewardStreamExperience
from RL.models.sac_model import SACModel
import numpy as np


class SafeSACTrainAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, sac_act_agent: SACActAgent, exp_buffer_agent: ExperienceBufferAgent):
        super().__init__(context, name)
        self.sac_act_agent = sac_act_agent
        self.experience_buffer_agent = exp_buffer_agent
        self.experience_buffer_agent.create_experiences = self.create_experiences
        self.num_safety_critics = len(self.context.safety_stream_names)
        self.num_normal_critics = self.context.num_critics - self.num_safety_critics
        assert self.num_safety_critics > 0, "No Safety Stream!"
        assert self.num_normal_critics > 0, "At least one normal critic is needed to optimize the actor"
        self.target_model = SACModel(context, "{0}/target_model".format(name), num_critics=self.context.num_critics + self.num_safety_critics)
        self.sac_act_agent.model.setup_training("{0}/training_ops".format(name))
        self.total_updates = 0

    def create_experiences(self):
        exps = []
        for i in range(self.context.num_envs):
            rewards = [self.runner.rewards[i]] + [self.runner.infos[i][stream_name + '_reward'] for stream_name in self.context.safety_stream_names]
            exps.append(MultiRewardStreamExperience(self.runner.prev_obss[i], self.runner.actions[i], rewards, self.runner.dones[i], self.runner.infos[i], self.runner.obss[i]))
        return exps

    def pre_act(self):
        if self.context.normalize_observations:
            self.sac_act_agent.model.update_states_running_stats(self.runner.obss)

    def get_target_network_V(self, states):
        target_actions, _, _, target_logpis = self.target_model.actions_means_logstds_logpis(states, noise=self.target_model.sample_actions_noise(len(states)))
        Q = np.min(np.array([self.target_model.Q(i, states, target_actions) for i in range(self.num_normal_critics)]), axis=0)
        assert len(Q) == len(states)
        assert len(target_logpis) == len(states)
        V = Q - self.context.alpha * target_logpis
        V_dashes = []
        for safety_stream_idx in range(self.num_safety_critics):
            V_dashes.append(self.target_model.Q(self.num_normal_critics + safety_stream_idx, states, target_actions))
        return V, V_dashes

    def train(self):
        c = self.context
        states, actions, multistream_rewards, dones, infos, next_states = self.experience_buffer_agent.experience_buffer.random_experiences_unzipped(c.minibatch_size)
        rewards = multistream_rewards[:, 0]  # for all sampled states, chose the primary reward
        safety_rewards = multistream_rewards[:, 1:]
        next_states_V, next_states_V_dashes = self.get_target_network_V(next_states)

        # TODO: implement TD error clipping

        # train main critics:
        desired_Q_values = rewards + (1 - dones.astype(np.int)) * (c.gamma ** c.nsteps) * next_states_V
        critic_loss = np.mean([self.sac_act_agent.model.train_critic(i, states, actions, desired_Q_values) for i in range(self.context.num_critics)])
        # train safety critics:
        safety_critic_loss = 0
        for safety_idx in range(self.num_safety_critics):
            desired_Q_dash_values = safety_rewards[: safety_idx] + (1 - dones.astype(np.int)) * (c.gamma ** c.nsteps) * next_states_V_dashes[safety_idx]
            safety_critic_loss += self.sac_act_agent.model.train_critic(self.num_normal_critics + safety_idx, states, actions, desired_Q_dash_values) / self.num_safety_critics
        # train actor:
        actor_loss, actor_critic_Q, actor_logstds, actor_logpis, actor_safety_critic_Qs = self.sac_act_agent.model.train_actor(states, self.sac_act_agent.model.sample_actions_noise(len(states)), self.context.alpha, self.context.safety_coeffs)
        return critic_loss, actor_loss, np.mean(actor_critic_Q), np.mean(actor_logstds), np.mean(actor_logpis), actor_safety_critic_Qs

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
