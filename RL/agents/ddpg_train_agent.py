import RL
from RL.agents.ddpg_act_agent import DDPGActAgent
from RL.agents.experience_buffer_agent import ExperienceBufferAgent
from RL.models.ddpg_model import DDPGModel
import numpy as np


class DDPGTrainAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, ddpg_act_agent: DDPGActAgent, exp_buffer_agent: ExperienceBufferAgent):
        super().__init__(context, name)
        self.ddpg_act_agent = ddpg_act_agent
        self.experience_buffer_agent = exp_buffer_agent
        self.target_model = DDPGModel(context, "{0}/target_model".format(name))
        self.ddpg_act_agent.model.setup_training("{0}/training_ops".format(name))

    def pre_act(self):
        self.ddpg_act_agent.model.update_states_running_stats([self.context.frame_obs])

    def get_target_network_V(self, states):
        actions = self.ddpg_act_agent.exploit_policy(self.ddpg_act_agent.model if self.context.double_dqn else self.target_model, states)
        Q = self.target_model.Q(states, actions)
        return Q

    def train(self):
        c = self.context
        states, actions, rewards, dones, infos, next_states = self.experience_buffer_agent.experience_buffer.random_experiences_unzipped(c.minibatch_size)
        next_states_V = self.get_target_network_V(next_states)
        desired_Q_values = rewards + (1 - dones.astype(np.int)) * (c.gamma ** c.nsteps) * next_states_V
        if c.clip_td_error:
            Q = self.ddpg_act_agent.model.Q(states, actions)
            td_errors = desired_Q_values - Q
            td_errors = np.clip(-1, 1)
            desired_Q_values = Q + td_errors
        self.ddpg_act_agent.model.train_critic(states, actions, desired_Q_values)
        self.ddpg_act_agent.model.train_actor(states)

    def post_act(self):
        c = self.context
        self.ddpg_act_agent.model.update_actions_running_stats([c.frame_action])
        if c.frame_id % c.train_every == 0 and c.frame_id >= c.minimum_experience:
            self.train()
