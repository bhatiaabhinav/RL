import RL
from RL.agents.dqn_act_agent import DQNActAgent
from RL.agents.experience_buffer_agent import ExperienceBufferAgent
from RL.common.experience_buffer import MultiRewardStreamExperience
from RL.models.dqn_model import Brain
import numpy as np


class DQNTrainAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, dqn_act_agent: DQNActAgent, exp_buffer_agent: ExperienceBufferAgent, loss_coeffs_per_head=[1.0]):
        super().__init__(context, name)
        self.dqn_act_agent = dqn_act_agent
        self.experience_buffer_agent = exp_buffer_agent
        self.experience_buffer_agent.create_experiences = self.create_experiences
        self.loss_coeffs = loss_coeffs_per_head
        assert len(self.loss_coeffs) == len(self.dqn_act_agent.head_names)
        if not hasattr(self.context.gamma, "__len__") and len(self.dqn_act_agent.head_names) > 1:
            RL.logger.warn("You are using same gamma for all reward streams. Are you sure this is intentional?")
        self.target_model = Brain(self.context, '{0}/target_brain'.format(self.name), True, head_names=self.dqn_act_agent.head_names)

    def pre_act(self):
        self.dqn_act_agent.model.update_running_stats(self.runner.obss)

    def optimize(self, states, actions, *desired_Q_values_per_head):
        all_rows = np.arange(len(states))
        Q_per_head = self.dqn_act_agent.model.get_Q(states)
        for Q, desired_Q_values in zip(Q_per_head, desired_Q_values_per_head):
            td_errors = desired_Q_values - Q[all_rows, actions]
            if self.context.clip_td_error:
                td_errors = np.clip(td_errors, -1, 1)
            Q[all_rows, actions] = Q[all_rows, actions] + td_errors
        combined_loss, Q_losses, Q_mpes = self.dqn_act_agent.model.train(states, *Q_per_head, loss_coeffs_per_head=self.loss_coeffs)
        return combined_loss, Q_losses, Q_mpes

    def get_target_network_V_per_head(self, states):
        all_rows = np.arange(self.context.minibatch_size)
        actions = self.dqn_act_agent.exploit_policy(self.dqn_act_agent.model if self.context.double_dqn else self.target_model, states)
        Q_per_head = self.target_model.get_Q(states)
        Vs = [Q[all_rows, actions] for Q in Q_per_head]
        return Vs

    def train(self):
        c = self.context
        states, actions, multistream_rewards, dones, infos, next_states = self.experience_buffer_agent.experience_buffer.random_experiences_unzipped(
            c.minibatch_size)
        next_states_V_per_head = self.get_target_network_V_per_head(next_states)
        desired_Q_values_per_head = []
        gamma = c.gamma
        if not hasattr(c.gamma, "__len__"):
            gamma = [gamma] * len(self.dqn_act_agent.head_names)
        for head_id in range(len(self.dqn_act_agent.head_names)):
            desired_Q_values = multistream_rewards[:, head_id] + (1 - dones.astype(np.int)) * (gamma[head_id] ** c.nsteps) * next_states_V_per_head[head_id]
            desired_Q_values_per_head.append(desired_Q_values)
        combined_loss, Q_losses, Q_mpes = self.optimize(states, actions, *desired_Q_values_per_head)

    def create_experiences(self):
        r = self.runner
        exps = []
        for i in range(self.context.num_envs):
            reward = [r.rewards[i]]
            exps.append(MultiRewardStreamExperience(r.prev_obss[i], r.actions[i], reward, r.dones[i], r.infos[i], r.obss[i]))
        return exps

    def post_act(self):
        r = self.runner
        c = self.context
        if r.step_id % c.train_every == 0 and r.step_id >= c.minimum_experience:
            self.train()
