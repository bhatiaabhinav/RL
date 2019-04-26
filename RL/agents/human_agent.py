import RL
import numpy as np


class SlowHuman(RL.Agent):
    def act(self):
        action = None
        while action is None:
            action = eval(input("Pick an action from space={0}: ".format(self.context.env.action_space)))
            action = np.array(action)
            if not self.context.env.action_space.contains(action):
                print("Illegal action. Pick again.")
                action = None
        return action
