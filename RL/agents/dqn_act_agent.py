import RL
from RL.models.dqn_model import Brain
import numpy as np


class DQNActAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, head_names=["default"]):
        super().__init__(context, name)
        self.head_names = head_names
        self.model = Brain(self.context, '{0}/main_brain'.format(self.name), False, head_names=self.head_names)

    def epsilon_greedy_policy(self, model: Brain, states, epsilon):
        if not hasattr(self, "_exp_pol_called") and len(self.head_names) > 1:
            RL.logger.log("There are multiple Q heads in agent {0}. Exploit policy will choose greedy action using first head only".format(self.name))
            self._exp_pol_called = True
        greedy_actions, = model.get_argmax_Q(states, head_names=[self.head_names[0]])
        r = np.random.random(size=[len(states)])
        greedy_mask = (r > epsilon).astype(np.int)
        random_actions = np.random.randint(0, self.context.env.action_space.n, size=[len(states)])
        actions = (1 - greedy_mask) * random_actions + greedy_mask * greedy_actions
        return actions

    def exploit_policy(self, model: Brain, states):
        return self.epsilon_greedy_policy(model, states, self.context.exploit_epsilon)

    def policy(self, model: Brain, states, exploit_modes):
        assert len(exploit_modes) == len(states)
        epsilon = [self.context.exploit_epsilon if m else self.context.epsilon for m in exploit_modes]
        return self.epsilon_greedy_policy(model, states, np.asarray(epsilon))

    def act(self):
        if not self.context.eval_mode and self.runner.num_steps < self.context.minimum_experience:
            return None  # let the random player act
        else:
            return self.policy(self.model, self.runner.obss, self.context.force_exploits)
