import RL
from RL.agents.simple_dqn_act_agent import SimpleDQNActAgent
from RL.agents.experience_buffer_agent import ExperienceBufferAgent
from RL.models.simple_dqn_model import Brain
import numpy as np


class SimpleDQNTrainAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, dqn_act_agent: SimpleDQNActAgent, exp_buffer_agent: ExperienceBufferAgent):
        super().__init__(context, name, True)
        self.dqn_act_agent = dqn_act_agent
        self.experience_buffer_agent = exp_buffer_agent
        self.target_model = Brain(self.context, '{0}/target_brain'.format(self.name), True)

    def pre_act(self):
        if self.context.normalize_observations:
            self.dqn_act_agent.model.update_running_stats(self.runner.obss)

    def optimize(self, states, actions, desired_Q_values):
        all_rows = np.arange(len(states))
        Q = self.dqn_act_agent.model.Q(states)
        td_errors = desired_Q_values - Q[all_rows, actions]
        if self.context.clip_td_error:
            td_errors = np.clip(td_errors, -1, 1)
        Q[all_rows, actions] = Q[all_rows, actions] + td_errors
        Q_loss, Q_mpe = self.dqn_act_agent.model.train(states, Q)
        return Q_loss, Q_mpe

    def get_target_network_V(self, states):
        all_rows = np.arange(self.context.minibatch_size)
        actions = self.dqn_act_agent.exploit_policy(self.dqn_act_agent.model if self.context.double_dqn else self.target_model, states)
        return self.target_model.Q(states)[all_rows, actions]

    def train(self):
        c = self.context
        states, actions, rewards, dones, infos, next_states = self.experience_buffer_agent.experience_buffer.random_experiences_unzipped(c.minibatch_size)
        next_states_V = self.get_target_network_V(next_states)
        desired_Q_values = rewards + (1 - dones.astype(np.int)) * (c.gamma ** c.nsteps) * next_states_V
        Q_loss, Q_mpe = self.optimize(states, actions, desired_Q_values)

    def post_act(self):
        r = self.runner
        c = self.context
        if r.step_id % c.train_every == 0 and r.step_id >= c.minimum_experience:
            for sgd_step_id in range(c.gradient_steps):
                self.train()
