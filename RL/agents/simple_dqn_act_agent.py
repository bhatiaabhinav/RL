import RL
from RL.models.simple_dqn_model import Brain
import numpy as np


class SimpleDQNActAgent(RL.Agent):
    '''supports single env thread only'''
    def __init__(self, context: RL.Context, name):
        super().__init__(context, name, False)
        self.model = Brain(self.context, '{0}/main_brain'.format(self.name), False)

    def greedy_policy(self, model: Brain, states):
        '''returns greedy actions for given states'''
        return self.model.argmax_Q(states)

    def epsilon_greedy_policy(self, model: Brain, states, epsilon):
        '''returns ep-greedy actions for given states and given epsilon'''
        greedy_actions = self.greedy_policy(self.model, states)
        r = np.random.random(size=[len(states)])
        greedy_mask = (r > epsilon).astype(np.int)
        random_actions = np.random.randint(0, self.context.env.action_space.n, size=[len(states)])
        actions = (1 - greedy_mask) * random_actions + greedy_mask * greedy_actions
        return actions

    def exploit_policy(self, model: Brain, states):
        '''returns ep-greedy actions for given states with exploit epsilon'''
        return self.epsilon_greedy_policy(model, states, self.context.exploit_epsilon)

    def policy(self, model: Brain, states):
        '''return ep-greedy actions for given states for appropriate epslion depending on exploit or explore mode (i.e. context.force_exploit)'''
        epsilon = self.context.exploit_epsilon if self.context.force_exploit else self.context.epsilon
        return self.epsilon_greedy_policy(model, states, epsilon)

    def act(self):
        if not self.context.eval_mode and self.runner.num_steps < self.context.minimum_experience:
            return None  # let the random player act
        else:
            return self.policy(self.model, [self.runner.obs])[0]
