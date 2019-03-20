import numpy as np
import gym


class RandomMDP(gym.Env):
    metadata = {
        'render.modes': ['ansi', 'human']
    }

    def __init__(self, n_states, n_actions, n_term_states, max_len=200, seed=None):
        self.action_space = gym.spaces.Discrete(n_actions)
        self.observation_space = gym.spaces.Discrete(n_states)

        assert n_states > n_term_states
        self._n_states = n_states
        self._n_non_term_states = n_states - \
            n_term_states  # these are non terminal states
        self._n_actions = n_actions
        self._n_term_states = n_term_states
        self._max_len = max_len

        # this is to be used for defining the env.
        self._random = np.random.RandomState(seed=seed)
        self._random_step = np.random.RandomState(
            seed=None)  # this for stepping through it

        # the axis semantics for the transition function are: s, a, s'
        self._state_trans_p = self._random.uniform(
            low=0, high=1, size=[self._n_non_term_states, self._n_actions, self._n_states])
        sum_s_dash = np.sum(self._state_trans_p, axis=-1, keepdims=True)
        self._state_trans_p = self._state_trans_p / sum_s_dash
        self._start_state_p = self._random.uniform(
            low=0, high=1, size=[self._n_non_term_states])
        self._start_state_p = self._start_state_p / np.sum(self._start_state_p)
        self._reward_fn = self._random.uniform(low=-1, high=1, size=[self._n_non_term_states, self._n_actions, self._n_states])
        self._reward_fn = np.clip(self._reward_fn * 4, -1, 1)  # constrast = 4
        self._safety_reward_fn = self._random.uniform(
            size=[self._n_non_term_states, self._n_actions, self._n_states])
        self._safety_reward_fn = np.clip(4 * (self._safety_reward_fn - 0.5) + 0.5, 0, 1)  # contrast = 4
        self._safety_reward_fn = -self._safety_reward_fn

        self.metadata["reward_fn"] = np.zeros(shape=[self._n_states, self._n_actions, self._n_states])
        self.metadata["reward_fn"][0:self._n_non_term_states, :, :] = self._reward_fn
        self.metadata["safety_reward_fn"] = np.zeros(shape=[self._n_states, self._n_actions, self._n_states])
        self.metadata["safety_reward_fn"][0:self._n_non_term_states, :, :] = self._safety_reward_fn
        self.metadata["transition_fn"] = np.zeros(shape=[self._n_states, self._n_actions, self._n_states])
        self.metadata["transition_fn"][0:self._n_non_term_states, :, :] = self._state_trans_p
        self.metadata["start_state_fn"] = np.zeros(shape=[self._n_states])
        self.metadata["start_state_fn"][0:self._n_non_term_states] = self._start_state_p
        if self._n_term_states > 0:
            self.metadata["reward_fn"][self._n_non_term_states:, :, :] = 0
            self.metadata["safety_reward_fn"][self._n_non_term_states:, :, :] = 0
            self.metadata["transition_fn"][self._n_non_term_states:, :, self._n_non_term_states:] = np.expand_dims(np.identity(self._n_term_states), axis=1)
            self.metadata["transition_fn"][self._n_non_term_states:, :, 0:self._n_non_term_states] = 0
            self.metadata["start_state_fn"][self._n_non_term_states:] = 0

        self._cur_state = None
        self._cur_action = None
        self._cur_reward = 0
        self._cur_safety_reward = 0
        self._cur_done = False
        self._steps = 0
        self.seed(None)

    def seed(self, seed):
        self._random_step.seed(seed)

    def reset(self):
        self._cur_state = self._random_step.choice(
            self._n_non_term_states, p=self._start_state_p)
        self._steps = 0
        self._cur_done = False
        self._cur_reward = 0
        self._cur_safety_reward = 0
        return self._cur_state

    def _is_terminal(self, state):
        return state >= self._n_non_term_states

    def step(self, action):
        assert action < self._n_actions, "Invalid action"
        assert self._cur_state is not None, "Was reset called?"
        assert not self._cur_done, "Was reset called?"

        next_state = self._random_step.choice(
            self._n_states, p=self._state_trans_p[self._cur_state, action])
        reward = self._reward_fn[self._cur_state, action, next_state]
        safety_reward = self._safety_reward_fn[self._cur_state, action, next_state]
        self._steps += 1
        done = self._is_terminal(next_state) or self._steps >= self._max_len
        info = {'Safety_reward': safety_reward}
        self._cur_state = next_state
        self._cur_action = action
        self._cur_reward = reward
        self._cur_safety_reward = safety_reward
        self._cur_done = done
        return self._cur_state, reward, done, info

    def render(self, mode='human'):
        text = "(action: {0})\n".format(self._cur_action)
        text += "(reward: {0})\n".format(self._cur_reward)
        text += "(safety_reward: {0})\n".format(self._cur_safety_reward)
        for i in range(self._n_states):
            if self._cur_state == i:
                text += "{0} <--\n".format(i)
            else:
                text += "{0}\n".format(i)
        if mode == 'human':
            print(text)
        return text

    def close(self):
        pass
