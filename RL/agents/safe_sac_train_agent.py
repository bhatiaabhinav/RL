import RL
from RL.agents.sac_act_agent import SACActAgent
from RL.agents.experience_buffer_agent import ExperienceBufferAgent
from RL.models.safe_sac_model import SafeSACModel
import numpy as np
import random


class SafeSACTrainAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, sac_act_agent: SACActAgent, exp_buffer_agent: ExperienceBufferAgent, exp_buff_agent_small: ExperienceBufferAgent):
        super().__init__(context, name)
        self.sac_act_agent = sac_act_agent
        self.experience_buffer_agent = exp_buffer_agent
        self.exp_buff_agent_small = exp_buff_agent_small
        self.target_model = SafeSACModel(context, "{0}/target_model".format(name), num_actors=0, num_critics=0, num_valuefns=2)
        self.sac_act_agent.model.setup_training("{0}/training_ops".format(name))
        self.total_updates = 0
        self.critic_ids = list(range(self.context.num_critics + 1))
        self.start_state_buff = []
        self.running_bias = 0  # diff between emperical on-policy cost and estimated on-policy cost
        self.jc_est = self.context.cost_threshold
        self.jc_est_last_n = 4

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

    def get_Q_targets_and_safety_Q_targets(self, rewards, costs, dones, next_states):
        c = self.context
        next_states_all_V = self.target_model.V([0, 1], next_states)
        next_states_V = next_states_all_V[0]
        next_states_safety_V = next_states_all_V[1]
        Q_targets = rewards + (1 - dones.astype(np.int)) * (c.gamma ** c.nsteps) * next_states_V
        safety_Q_targets = costs + (1 - dones.astype(np.int)) * (c.cost_gamma ** c.nsteps) * next_states_safety_V
        return Q_targets, safety_Q_targets

    def train(self):
        c = self.context
        states, actions, rewards, costs, dones, infos, next_states = self.experience_buffer_agent.experience_buffer.random_experiences_unzipped(c.minibatch_size, return_costs=True)
        assert np.all(costs >= 0), "How are the costs negative?"
        noise = self.sac_act_agent.model.sample_actions_noise(len(states))
        V_targets, safety_V_targets = self.get_V_and_safety_V_targets(states, noise)
        Q_targets, safety_Q_targets = self.get_Q_targets_and_safety_Q_targets(rewards, costs, dones, next_states)
        Q_targets_per_critic = [Q_targets] * c.num_critics
        # TODO: implement TD error clipping
        valuefn_loss = self.sac_act_agent.model.train_valuefns(states, [0], [V_targets], [1])
        safety_valuefn_loss = self.sac_act_agent.model.train_valuefns(states, [1], [safety_V_targets], [1])
        critic_loss = self.sac_act_agent.model.train_critics(states, actions, self.critic_ids[:c.num_critics], Q_targets_per_critic, [1 / c.num_critics] * c.num_critics)
        safety_critic_loss = self.sac_act_agent.model.train_critics(states, actions, [c.num_critics], [safety_Q_targets], [1])

        # uncomment bottom two lines (and comment first two lines) to swtich to an unbiased estimate of Jc:
        # Jc_est = self.Jc_est(noise)  # this is a biased estimate. But more readily available.
        # on_policy_cost = Jc_est + self.running_bias  # less biased estimate

        if self.runner.episode_step_id == 0:
            data = RL.stats.get('Env-0 Episode Cost')[-self.jc_est_last_n:]
            self.jc_est = np.mean(data)
            se = np.std(data, ddof=1) / np.sqrt(self.jc_est_last_n)
            if 2 * se > self.jc_est or se == 0:
                self.jc_est_last_n += 2
            else:
                self.jc_est_last_n -= 2
            self.jc_est_last_n = int(np.clip(self.jc_est_last_n, 4, min(self.runner.num_episodes, 50)))
            RL.logger.debug("n, jc±se: ", str(self.jc_est_last_n), ", ", str(self.jc_est), "±", str(np.round(se, 4)))

        # uncomment these two lines to change from hybrid-off-policy to uu-on-policy gradient:
        # recent_states = self.exp_buff_agent_small.experience_buffer.random_states(c.minibatch_size)
        # states = recent_states

        actor_loss, actor_critics_Qs, actor_logstds, actor_logpis = self.sac_act_agent.model.train_actor(states, noise, [0, c.num_critics], [1, 1], self.jc_est, c.alpha, c._lambda_scale / (c.cost_threshold * c.cost_scaling), c.cost_threshold * c.cost_scaling)

        return valuefn_loss, safety_valuefn_loss, critic_loss, safety_critic_loss, actor_loss, np.mean(actor_critics_Qs[0]), np.mean(actor_critics_Qs[1]), np.mean(actor_logstds), np.mean(actor_logpis), self.jc_est

    def post_act(self):
        # print(self.runner.cost)
        # if self.runner.episode_step_id == 0:
        #     self.start_state_buff.append(self.runner.obs)
        c = self.context
        if self.context.normalize_actions:
            self.sac_act_agent.model.update_actions_running_stats(self.runner.actions)
        if self.runner.step_id % c.train_every == 0 and self.runner.step_id >= c.minimum_experience:
            valuefn_loss, safety_valuefn_loss, critic_loss, safety_critic_loss, actor_loss, av_actor_critic_Q, av_actor_critic_safety_Q, av_logstd, av_logpi, Jc_est = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            for sgd_step_id in range(self.context.gradient_steps):
                valuefn_loss, safety_valuefn_loss, critic_loss, safety_critic_loss, actor_loss, av_actor_critic_Q, av_actor_critic_safety_Q, av_logstd, av_logpi, Jc_est = self.train()
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
                RL.stats.record("Average Start States Cost V", Jc_est)
                RL.stats.record("Average Action LogStd", av_logstd)
                RL.stats.record("Average Action LogPi", av_logpi)

    def Jc_est(self, noise):
        size = min(self.context.minibatch_size, len(self.start_state_buff))
        start_states = np.asarray(random.sample(self.start_state_buff, size))
        noise = noise[0:size]
        start_states_actions = self.sac_act_agent.model.actions(start_states, noise=noise)
        Jc_est = np.mean(self.sac_act_agent.model.Q([self.context.num_critics], start_states, start_states_actions)[0])
        return Jc_est

    def post_episode(self):
        # if self.runner.step_id >= self.context.minimum_experience:minion_policy_costimum_experience
        #     G = RL.stats.get('Env-0 Episode Cost')[-1]
        #     noise = self.sac_act_agent.model.sample_actions_noise(self.context.minibatch_size)
        #     self.running_bias = 0.9 * self.running_bias + 0.1 * (G - self.Jc_est(noise))
        # if self.jc_est is not None:
        #     RL.stats.record("Running Jc Est Bias", self.jc_est)
        # self.jc_est = None
        pass
