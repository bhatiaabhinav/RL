import os
from typing import List

import gym
import numpy as np

from RL.common import logger
from RL.common.context import (Agent, Context, PygletLoop, RandomPlayAgent,  # noqa: F401
                               RLRunner, SeedingAgent, SimpleRenderingAgent)
from RL.common.experience_buffer import (Experience, ExperienceBuffer,
                                         MultiRewardStreamExperience)
from RL.common.plot_renderer import BasePlotRenderer
from RL.dqn.safe_dqn import SafetyStatsRecorderAgent
from RL.common.random_mdp import RandomMDP

class QTableBrain:
    def __init__(self, context: Context, name):
        self.context = context
        self.name = name
        self.Q_updates = 0
        self.n_states = self.context.env.observation_space.n
        self.n_actions = self.context.env.action_space.n
        self._Q = self._create_Q()

    def _create_Q(self):
        return 2e-3 * np.random.sample(size=[self.n_states, self.n_actions]) - 1e-3

    def get_Q(self, states, actions=None):
        if actions is None:
            return self._Q[np.asarray(states), :]
        else:
            return self._Q[np.asarray(states), np.asarray(actions)]

    def get_argmaxQ(self, states):
        return np.argmax(self.get_Q(states), axis=-1)

    def get_maxQ(self, states):
        return np.max(self.get_Q(states), axis=-1)

    def _Q_training_step(self, states, error):
        self._Q[states, :] = self._Q[states, :] + \
            self.context.learning_rate * error

    def train(self, states, desired_Q):
        self.Q_updates += 1
        Q = self.get_Q(states)
        error = desired_Q - Q
        abs_error = np.abs(error)
        Q_loss = np.mean(abs_error)
        pe = 100 * abs_error / (np.abs(desired_Q) + 1e-3)
        Q_mpe = np.mean(pe)
        V = np.max(Q, axis=-1)
        self._Q_training_step(states, error)
        mb_av_V_summary_name = "{0}/mb_av_V".format(self.name)
        Q_loss_summary_name = "{0}/Q_loss".format(self.name)
        Q_mpe_summary_name = "{0}/Q_mpe".format(self.name)
        if self.Q_updates == 1:
            self.context.summaries.setup_scalar_summaries(
                [mb_av_V_summary_name, Q_loss_summary_name, Q_mpe_summary_name])
        if self.Q_updates % 10 == 0:
            self.context.log_summary({mb_av_V_summary_name: np.mean(
                V), Q_loss_summary_name: Q_loss, Q_mpe_summary_name: Q_mpe}, self.context.frame_id)
        return Q_loss, Q_mpe


class QAgent(Agent):
    def __init__(self, context: Context, name, head_names=["default"]):
        super().__init__(context, name)
        self.head_names = head_names
        if not hasattr(self.context.gamma, "__len__") and len(head_names) > 1:
            logger.warn(
                "You are using same gamma for all reward streams. Are you sure this is intentional?")
        self.create_brains()
        self.create_experience_buffer()

    def create_brains(self):
        self.brains = []  # List[QTableBrain]
        for head_name in self.head_names:
            self.brains.append(QTableBrain(
                self.context, self.name + "/" + head_name))

    def create_experience_buffer(self):
        if self.context.experience_buffer_megabytes is None:
            self.experience_buffer = ExperienceBuffer(
                length=self.context.experience_buffer_length)
        else:
            self.experience_buffer = ExperienceBuffer(
                size_in_bytes=self.context.experience_buffer_megabytes * (1024**2))
        self.nstep_buffer = []  # type: List[Experience]

    def add_to_experience_buffer(self, exp: MultiRewardStreamExperience):
        reward_to_prop_back = exp.reward
        for old_exp in reversed(self.nstep_buffer):
            if old_exp.done:
                break
            reward_to_prop_back = np.asarray(
                self.context.gamma) * reward_to_prop_back
            old_exp.reward += reward_to_prop_back
            old_exp.next_state = exp.next_state
            old_exp.done = exp.done
        self.nstep_buffer.append(exp)
        if len(self.nstep_buffer) >= self.context.nsteps:
            self.experience_buffer.add(self.nstep_buffer.pop(0))
            assert len(self.nstep_buffer) == self.context.nsteps - 1

    def exploit_policy(self, states):
        if not hasattr(self, "_exp_pol_called") and len(self.head_names) > 1:
            logger.log(
                "There are multiple Q heads in agent {0}. Exploit policy will choose greedy action using first head only".format(self.name))
            self._exp_pol_called = True
        greedy_actions = self.brains[0].get_argmaxQ(states)
        return greedy_actions

    def policy(self, states):
        r = np.random.random(size=[len(states)])
        exploit_mask = (r > self.context.epsilon).astype(np.int)
        exploit_actions = self.exploit_policy(states)
        explore_actions = np.random.randint(
            0, self.context.env.action_space.n, size=[len(states)])
        actions = (1 - exploit_mask) * explore_actions + \
            exploit_mask * exploit_actions
        return actions

    def act(self):
        return self.policy([self.context.frame_obs])[0]
        # r = np.random.random()
        # if r > self.context.epsilon:
        #     return np.argmax(self.brains[0]._Q[self.context.frame_obs, :])
        # else:
        #     return self.context.env.action_space.sample()

    def get_all_stream_rewards_current_frame(self):
        return [self.context.frame_reward]

    def get_target_V_per_head(self, states):
        all_rows = np.arange(len(states))
        actions = self.exploit_policy(states)
        Q_per_head = [b.get_Q(states) for b in self.brains]
        Vs = [Q[all_rows, actions] for Q in Q_per_head]
        return Vs

    def optimize(self, states, actions, *desired_Q_values_per_head):
        all_rows = np.arange(len(states))
        for brain, desired_Q_values in zip(self.brains, desired_Q_values_per_head):
            Q = brain.get_Q(states)
            td_errors = desired_Q_values - Q[all_rows, actions]
            if self.context.clip_td_error:
                td_errors = np.clip(td_errors, -1, 1)
            Q[all_rows, actions] = Q[all_rows, actions] + td_errors
            brain.train(states, Q)

    def train(self):
        c = self.context
        states, actions, multistream_rewards, dones, infos, next_states = self.experience_buffer.random_experiences_unzipped(
            c.minibatch_size)
        next_states_V_per_head = self.get_target_V_per_head(next_states)
        desired_Q_values_per_head = []
        gamma = c.gamma
        if not hasattr(c.gamma, "__len__"):
            gamma = [gamma] * len(self.head_names)
        for head_id in range(len(self.head_names)):
            desired_Q_values = multistream_rewards[:, head_id] + (1 - dones.astype(
                np.int)) * (gamma[head_id] ** c.nsteps) * next_states_V_per_head[head_id]
            desired_Q_values_per_head.append(desired_Q_values)
        self.optimize(states, actions, *desired_Q_values_per_head)

    def post_act(self):
        c = self.context
        if self.context.eval_mode:
            return
        reward = self.get_all_stream_rewards_current_frame()
        self.add_to_experience_buffer(MultiRewardStreamExperience(
            c.frame_obs, c.frame_action, reward, c.frame_done, c.frame_info, c.frame_obs_next))

        if c.frame_id % c.train_every == 0 and c.frame_id >= c.minimum_experience - 1:
            self.train()
            # q_cur = self.brains[0]._Q[c.frame_obs, c.frame_action]
            # q_target = c.frame_reward + c.gamma * np.max(self.brains[0]._Q[c.frame_obs_next, :])
            # self.brains[0]._Q[c.frame_obs, c.frame_action] += c.learning_rate * (q_target - q_cur)


class SafeQAgent(QAgent):
    def __init__(self, context: Context, name):
        head_names = ['default'] + context.safety_stream_names
        super().__init__(context, name, head_names=head_names)

    def _reset_feasible_actions_count_stats(self):
        self.av_feasible_actions_count = 0
        self.nonrandom_actions_count = 0

    def _update_feasible_actions_count_stats(self, feasible_action_count):
        self.av_feasible_actions_count = (
            self.av_feasible_actions_count * self.nonrandom_actions_count + feasible_action_count) / (self.nonrandom_actions_count + 1)
        self.nonrandom_actions_count += 1

    def pre_episode(self):
        super().pre_episode()
        self._reset_feasible_actions_count_stats()

    def get_all_stream_rewards_current_frame(self):
        rewards = [self.context.frame_reward]
        for stream_name in self.context.safety_stream_names:
            rewards.append(self.context.frame_info[stream_name + "_reward"])
        return rewards

    def exploit_policy(self, states, dont_record_stats=False):
        assert len(
            self.head_names) == 2, "Right now only one safety stream supported"
        Q, safety_Q = self.brains[0].get_Q(
            states), self.brains[1].get_Q(states)
        t = self.context.safety_threshold
        feasibility_mask = (safety_Q > t).astype(np.int)
        feasible_available_mask = np.any(
            feasibility_mask, axis=-1).astype(np.int)
        # additive mask should have -inf for infeasible actions and 0 for feasible:
        additive_mask = (1 - feasibility_mask) * -1e6
        Q_masked = Q + additive_mask
        action_greedy_feasible = np.argmax(Q_masked, axis=-1)
        action_safest = np.argmax(safety_Q, axis=-1)
        action = feasible_available_mask * action_greedy_feasible + \
            (1 - feasible_available_mask) * action_safest
        if not dont_record_stats:
            self._update_feasible_actions_count_stats(np.sum(feasibility_mask))
            self.safety_threshold = t
        return action

    def post_episode(self):
        super().post_episode()
        av_feasible_actions_count_summary_name = "av_feasible_actions_count"
        if self.context.episode_id == 0:
            self.context.summaries.setup_scalar_summaries(
                [av_feasible_actions_count_summary_name])
        self.context.log_summary(
            {av_feasible_actions_count_summary_name: self.av_feasible_actions_count}, self.context.episode_id)


class AdaptivePenaltySafeQAgent(QAgent):
    pass
        

class QTablePlotAgent(Agent):
    def __init__(self, context: Context, name, qAgent: QAgent, head_ids=0):
        super().__init__(context, name)
        self.qAgent = qAgent
        self.head_ids = head_ids
        self.plot = None  # type: BasePlotRenderer

    def start(self):
        if self.context.plot_Q:
            try:
                self.plot = BasePlotRenderer(window_caption=self.context.env_id + ":" +
                                             self.context.experiment_name + ":" + self.name, auto_dispatch_on_render=False)
            except Exception as e:
                logger.error(
                    "{0}: Could not create plot window. Reason = {1}".format(self.name, str(e)))

    def update(self):
        if self.plot and self.context.plot_Q and self.context.episode_id % self.context.render_interval == 0:
            self.plot.fig.clear()
            axess = self.plot.fig.subplots(1, len(self.head_ids), sharey=True)
            if not hasattr(axess, "__len__"):
                axess = [axess]
            self.plot.fig.suptitle("Q Table Visualization for {0} : Episode {1}".format(
                self.qAgent.name, self.context.episode_id))
            for head_id, axes in zip(self.head_ids, axess):
                Q = self.qAgent.brains[head_id]._Q
                axes.matshow(Q)
                axes.set_title("{0}".format(self.qAgent.head_names[head_id]))
            self.plot.render()

    def pre_episode(self):
        if self.plot:
            self.update()

    def post_episode(self):
        if self.plot and self.context.plot_Q and self.context.episode_id % self.context.render_interval == 0:
            filename = os.path.join(logger.get_dir(
            ), "Plots", self.name, "Q_Episode_{0}.png".format(self.context.episode_id))
            self.plot.save(path=filename)

    def post_act(self):
        self.update()

    def close(self):
        if self.plot:
            self.plot.close()


class FrozenLakeSafetyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        self.steps = 0
        return super().reset()

    def step(self, action):
        obs, r, d, info = super().step(action)
        self.steps += 1
        sr = -1 if (d and self.steps < 200 and r < 0.5) else 0
        info["Safety_reward"] = sr + info.get("Safety_reward", 0)
        return obs, r, d, info


class MyContext(Context):
    def make_env(self, env_id):
        if 'RandomMDP' in env_id:
            args = env_id.split('-')
            try:
                n_states, n_actions, n_terms, seed = eval(args[1]), eval(args[2]), eval(args[3]), eval(args[4])
                env = RandomMDP(n_states, n_actions, n_terms, seed=seed)
                return env
            except Exception as e:
                logger.error('Invalid env_id {0}. Error={1}'.format(env_id, e))
                raise e
        else:
            return super().make_env(env_id)

    def wrappers(self, env: gym.Env):
        env.metadata
        assert isinstance(
            env.observation_space, gym.spaces.Discrete), "Tabular Q learning cannot handle images or continuous state spaces"
        if 'FrozenLake8x8' in self.env_id:
            env = FrozenLakeSafetyWrapper(env)
        return env


def synchronus_Q_learning(transition_fn: np.ndarray, reward_fn: np.ndarray, gamma=0.99, seed=None, iterations=1000):
    plotter = BasePlotRenderer(window_caption="Sync Q Learning", auto_dispatch_on_render=True)

    def update(iter_no, Qs):
        plotter.fig.clear()
        plotter.fig.suptitle("Iteration {0}".format(iter_no))
        axess = plotter.fig.subplots(1, len(Qs), sharey=True)
        axess = [axess] if not hasattr(axess, "__len__") else axess
        for i, (axes, Q) in enumerate(zip(axess, Qs)):
            axes.set_title("Q{0}".format(i))
            axes.matshow(Q)
        plotter.render()

    n_states = transition_fn.shape[0]
    n_actions = transition_fn.shape[1]
    p = transition_fn
    r = reward_fn
    random = np.random.RandomState(seed=seed)
    Q = random.standard_normal(size=[n_states, n_actions])
    all_rows = np.arange(n_states)

    def policy():
        return np.argmax(Q, axis=-1)

    for iter_no in range(iterations):
        Q = np.sum(p * (r + gamma * Q[all_rows, policy()]), axis=-1)
        if iter_no % 10 == 0:
            update(iter_no, [Q])

    plotter.close()
    return Q


def synchronus_Safe_Q_learning(transition_fn: np.ndarray, reward_fn: np.ndarray, safety_reward_fn: np.ndarray, thres, gamma=0.99, seed=None, iterations=1000):
    plotter = BasePlotRenderer(window_caption="Sync Q Learning", auto_dispatch_on_render=True)
    n_states = transition_fn.shape[0]
    n_actions = transition_fn.shape[1]
    p = transition_fn
    r = reward_fn
    sr = safety_reward_fn
    random = np.random.RandomState(seed=seed)
    Q = random.standard_normal(size=[n_states, n_actions])
    Q_safe = random.standard_normal(size=[n_states, n_actions])
    Q_TD = None
    Q_safe_TD = None
    list_stats = []
    all_rows = np.arange(n_states)

    def policy():
        feasibility_mask = (Q_safe > thres).astype(np.int)
        feasible_available_mask = np.any(feasibility_mask, axis=-1).astype(np.int)
        # additive mask should have -inf for infeasible actions and 0 for feasible:
        additive_mask = (1 - feasibility_mask) * -1e6
        Q_masked = Q + additive_mask
        action_greedy_feasible = np.argmax(Q_masked, axis=-1)
        action_safest = np.argmax(Q_safe, axis=-1)
        action = feasible_available_mask * action_greedy_feasible + \
            (1 - feasible_available_mask) * action_safest
        return action
    
    def stats():
        pi = policy()
        V = Q[all_rows, pi]
        V_safe = Q_safe[all_rows, pi]
        av_feasible_actions = np.average(np.sum((Q_safe > thres).astype(np.int), -1))
        start_dist = c.env.metadata["start_state_fn"]
        J, J_safe = np.sum(start_dist * V), np.sum(start_dist * V_safe)
        error, error_safe = np.average(np.abs(Q_TD)), np.average(np.abs(Q_safe_TD))
        return {
            "av_V": np.average(V),
            "worst_V": np.min(V),
            "av_V_safe": np.average(V_safe),
            "worst_V_safe": np.min(V_safe),
            "J": J,
            "J_safe": J_safe,
            "error": error,
            "error_safe": error_safe,
            "av_feasible_actions": av_feasible_actions,
            "thres": thres
        }

    def get_curve(key):
        return [s[key] for s in list_stats]

    def update_graph():
        plotter.fig.clear()
        plotter.fig.suptitle("Safe Q Learning for {0}: Iteration {1}".format(c.env_id, len(list_stats)))
        axess = plotter.fig.subplots(2, 3, squeeze=False)
        axess[0, 0].set_title("Q Table")
        axess[0, 0].matshow(Q)
        axess[1, 0].set_title("Safety Q Table")
        axess[1, 0].matshow(Q_safe)
        x = np.arange(1, len(list_stats) + 1)
        axess[0, 1].set_title('Values per Iteration')
        axess[0, 1].plot(x, get_curve('J'), 'b-', x, get_curve('av_V'), 'g-', x, get_curve('worst_V'), 'y-', x, get_curve('error'), 'r-')
        axess[0, 1].legend(['J', 'Av V', 'Min V', 'Av TD Error'])
        axess[1, 1].set_title('Safety Values per Iteration')
        axess[1, 1].plot(x, get_curve('J_safe'), 'b-', x, get_curve('av_V_safe'), 'g-', x, get_curve('worst_V_safe'), 'y-', x, get_curve('error_safe'), 'r-', x, get_curve('thres'), 'c--')
        axess[1, 1].legend(['J', 'Av V', 'Min V', 'Av TD Error', 'Safety Thresh'])
        axess[1, 2].set_title('Av Feasible Actions per Iteration')
        axess[1, 2].plot(x, get_curve('av_feasible_actions'), 'm-')
        if c.render:
            plotter.render()

    for iter_no in range(iterations):
        mu = policy()
        Q_TD = np.sum(p * (r + gamma * Q[all_rows, mu]), axis=-1) - Q
        Q_safe_TD = np.sum(p * (sr + gamma * Q_safe[all_rows, mu]), axis=-1) - Q_safe
        Q, Q_safe = Q + Q_TD, Q_safe + Q_safe_TD
        list_stats.append(stats())
        if iter_no == 0 or (iter_no + 1) % c.render_interval == 0:
            update_graph()

    plotter.close()
    plotter.save(path=os.path.join(logger.get_dir(), "Training_Curves", "Thres_{0}.png".format(thres)))

    return Q, Q_safe, stats()


c = MyContext()
env = c.env
logger.info("Env: ", str(env.metadata))
# Q = synchronus_Q_learning(env.metadata["transition_fn"], env.metadata["reward_fn"], gamma=c.gamma, seed=c.seed, iterations=c.n_episodes)
Q, Q_safe, stats = synchronus_Safe_Q_learning(env.metadata["transition_fn"], env.metadata["reward_fn"], env.metadata["safety_reward_fn"], c.safety_threshold, gamma=c.gamma, seed=c.seed, iterations=c.n_episodes)
logger.log("Q: ", str(Q))
logger.log("Q_safe: ", str(Q_safe))
logger.log(str(stats))
# for thres in np.linspace(-2, 0, num=20):
#     Q, Q_safe, stats = synchronus_Safe_Q_learning(env.metadata["transition_fn"], env.metadata["reward_fn"], env.metadata["safety_reward_fn"], thres, gamma=c.gamma, seed=c.seed, iterations=c.n_episodes)
#     logger.log(str(stats))

# if __name__ == '__main__':
#     context = MyContext()

#     runner = RLRunner(context)

#     runner.register_agent(SeedingAgent(context, "seeder"))
#     # runner.register_agent(RandomPlayAgent(context, "RandomPlayer"))
#     if context.render:
#         runner.register_agent(SimpleRenderingAgent(context, "Video"))
#     q_agent = runner.register_agent(SafeQAgent(context, "Q"))
#     runner.register_agent(SafetyStatsRecorderAgent(context, "SafetyStats"))
#     if context.plot_Q:
#         runner.register_agent(QTablePlotAgent(
#             context, "Q", q_agent, head_ids=list(range(len(q_agent.head_names)))))
#     runner.register_agent(PygletLoop(context, "PygletLoop"))
#     runner.run()

#     context.close()
