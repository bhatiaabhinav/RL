import os
import sys
import pyglet
from datetime import timedelta
from typing import List  # noqa: F401

import gym
import numpy as np
import tensorflow as tf

from RL.common import logger
from RL.common.utils import set_global_seeds, ImagePygletWingow
from RL.common.atari_wrappers import wrap_atari  # noqa: F401,F403
from RL.common.summaries import Summaries


class Context:
    seed = 0
    gamma = 0.99
    default_env_id = 'CartPole-v0'
    default_experiment_name = 'untitled'
    eval_mode = False
    convs = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    states_embedding_hidden_layers = []
    Q_hidden_layers = [512]  # hidden layers
    n_episodes = 10000
    experience_buffer_length = 100000
    experience_buffer_megabytes = None
    minimum_experience = 20000  # in frames, before learning can begin
    final_epsilon = 0.01  # exploration rate in epislon-greedy action selection
    eval_epsilon = 0.001
    epsilon_anneal_over = 50000
    train_every = 4  # one sgd step every this many frames
    # target network updated every this many frames
    target_network_update_every = 4
    target_network_update_tau = 0.001  # ddpg style target network (soft) update step size
    minibatch_size = 32
    nsteps = 3  # n-step TD learning
    learning_rate = 6.25e-5
    double_dqn = False
    dueling_dqn = False
    l2_reg = 0
    clip_gradients = False
    clip_td_error = False
    video_interval = 50  # every these many episodes, video recording will be done
    save_every = 100  # every these many episodes, save a model
    exploit_every = 4  # every these many episodes, agent will play without exploration
    rnd_mode = False  # whether to use rnd intrinsic reward system
    rnd_num_features = 64  # predict these many random features of obs
    rnd_layers = [128]  # architecture of the rnd_predictor network
    rnd_learning_rate = 1e-4  # learning rate for the rnd_predictor network
    render = False
    sensitivity_visualizer = False
    plot_Q = False
    render_interval = 1
    render_vsync = False
    pyglet_fps = -1
    safety_threshold = -0.1
    penalty_safe_dqn_multiplier = 10
    penalty_safe_dqn_mode = False
    safety_stream_names = ['Safety']
    load_model_dir = None
    atari_framestack_k = 4
    atari_clip_rewards = True
    atari_episode_life = True
    lunar_no_crash_penalty_main_stream = False

    def __init__(self):
        self.env_id = self.default_env_id
        self.experiment_name = self.default_experiment_name
        self.summaries = None  # type: Summaries
        self.episode_id = 0
        self.frame_id = 0
        self.frame_obs = None
        self.frame_action = None
        self.frame_done = False
        self.frame_reward = 0
        self.frame_obs_next = None
        self.frame_info = {}
        self.intrinsic_returns = []
        self._read_args()
        logger.set_logdir(os.getenv('OPENAI_LOGDIR'), self.env_id, self.experiment_name)
        with open(os.path.join(logger.get_dir(), "context_class.json"), "w") as f:
            v = vars(Context)
            f.writelines(str(v))
        with open(os.path.join(logger.get_dir(), "context.json"), "w") as f:
            v = vars(self)
            f.writelines(str(v))
        logger.info("env_id is ", self.env_id)
        if self.experiment_name == self.default_experiment_name:
            logger.warn("No experiment_name argument was provided. It is highly recommened that you name your experiments")
        self.logdir = logger.get_dir()
        self.env = gym.make(self.env_id)
        self.env = self.wrappers(self.env)
        self.env = gym.wrappers.Monitor(self.env, os.path.join(self.logdir, 'monitor'), force=True, video_callable=self._video_schedule)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

    def _read_args(self):
        for arg in sys.argv[1:]:
            words = arg.split('=')
            assert len(
                words) == 2, "Invalid argument {0}. The format is --arg_name=arg_value".format(arg)
            assert words[0].startswith(
                '--'), "Invalid argument {0}. The format is --arg_name=arg_value".format(arg)
            arg_name = words[0][2:]
            try:
                arg_value = eval(words[1])
            except Exception:
                arg_value = words[1]
            assert hasattr(self, arg_name), "Unrecognized argument {0}".format(arg_name)
            setattr(self, arg_name, arg_value)

    def close(self):
        self.session.close()
        self.env.close()

    def wrappers(self, env):
        return env

    def _video_schedule(self, episode_id):
        video_interval = self.video_interval
        from gym.wrappers.monitor import capped_cubic_video_schedule
        if video_interval is not None and video_interval <= 0:
            return False
        if episode_id == self.n_episodes - 1:
            return True
        if video_interval is None:
            return capped_cubic_video_schedule(episode_id)
        return episode_id % video_interval == 0

    def should_eval_episode(self, episode_id=None):
        if episode_id is None:
            episode_id = self.episode_id
        return self.eval_mode or (self.exploit_every is not None and episode_id % self.exploit_every == 0)

    def activation_fn(self, x):
        return tf.nn.relu(x)

    @property
    def need_conv_net(self):
        return isinstance(self.env.observation_space, gym.spaces.Box) and len(self.env.observation_space.shape) >= 2

    @property
    def epsilon(self):
        if self.should_eval_episode():
            return self.eval_epsilon
        if self.total_steps < self.minimum_experience:
            return 1
        else:
            steps = self.total_steps - self.minimum_experience
            return self.final_epsilon + (1 - self.final_epsilon) * (1 - min(steps / self.epsilon_anneal_over, 1))

    @property
    def total_episodes(self):
        return len(self.env.stats_recorder.episode_rewards)

    @property
    def total_steps(self):
        return self.env.stats_recorder.total_steps

    @property
    def episode_steps(self):
        return self.env.stats_recorder.steps

    def log_stats(self, average_over=100, end='\n'):
        np.set_printoptions(precision=2)
        ep_id = self.episode_id
        reward = self.env.stats_recorder.episode_rewards[-1]
        length = self.env.stats_recorder.episode_lengths[-1]
        ep_type = self.env.stats_recorder.episode_types[-1]
        av_reward = np.mean(
            self.env.stats_recorder.episode_rewards[-average_over:])
        eval_ep_rewards = [tup[1] for tup in filter(lambda tup: tup[0] == 'e', zip(
            self.env.stats_recorder.episode_types[-average_over:], self.env.stats_recorder.episode_rewards[-average_over:]))]
        t = self.env.stats_recorder.timestamps[-1] - \
            self.env.stats_recorder.initial_reset_timestamp
        t = timedelta(seconds=int(t))
        eval_av_reward = np.mean(eval_ep_rewards) if len(
            eval_ep_rewards) > 0 else 0
        # intR = self.intrinsic_returns[-1]
        # av_intR = np.mean(self.intrinsic_returns[-average_over:])
        logger.log('{0} : Ep_id: {1:04d}\tR: {2:05.1f} {3}\tAv_R: {4:06.2f}\tAv_Eval_R: {5:06.2f}\tL: {6:03d}\tep: {7:4.2f}{8}'.format(
            str(t), ep_id, reward, ep_type, av_reward, eval_av_reward, length, self.epsilon, '\r' if end == '\r' else ''))
        if ep_id == 0:
            self.summaries.setup_scalar_summaries(["R", "R_exploit", "L", "epsilon"])
        summmary = {"R": reward, "L": length, "epsilon": self.epsilon}
        if ep_type == 'e':
            summmary["R_exploit"] = eval_ep_rewards[-1]
        self.log_summary(summmary, ep_id)

    def log_summary(self, kvs, step):
        self.summaries.write_summaries(kvs, step)


class Agent:
    def __init__(self, context: Context, name):
        self.context = context  # type: Context
        self.name = name
        self.runner = None  # type: RLRunner

    def start(self):
        pass

    def pre_episode(self):
        pass

    def pre_act(self):
        pass

    def act(self):
        return None

    def post_act(self):
        pass

    def post_episode(self):
        pass

    def close(self):
        pass


class PygletLoop(Agent):
    def __init__(self, context: Context, name):
        super().__init__(context, name)
        if self.context.pyglet_fps > 0:
            pyglet.clock.set_fps_limit(self.context.pyglet_fps)

    def post_act(self):
        pyglet.clock.tick()
        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event('on_draw')
            window.flip()


class SimpleRenderingAgent(Agent):
    def __init__(self, context: Context, name):
        super().__init__(context, name)
        self.vsync = context.render_vsync
        self.window = None

    def render(self):
        if self.window and self.context.render and self.context.episode_id % self.context.render_interval == 0:
            self.window.set_image(self.context.env.render(mode='rgb_array'))

    def start(self):
        try:
            if self.context.render:
                self.window = ImagePygletWingow(caption=self.context.env_id + ":" + self.context.experiment_name + ":" + self.name, vsync=self.vsync)
        except Exception as e:
            logger.error("{0}: Could not create window. Reason = {1}".format(self.name, str(e)))

    def pre_episode(self):
        self.render()

    def post_act(self):
        self.render()

    def close(self):
        if self.window:
            self.window.close()


class SeedingAgent(Agent):
    def __init__(self, context: Context, name):
        super().__init__(context, name)
        set_global_seeds(self.context.seed)

    def start(self):
        self.context.env.seed(self.context.seed)


class RandomPlayAgent(Agent):
    def act(self):
        return self.context.env.action_space.sample()


class RLRunner:
    def __init__(self, context: Context):
        self.agents = []  # type: List[Agent]
        self.context = context
        self.env = context.env
        self._agent_name_to_agent_map = {}
        self._closed = False

    def register_agent(self, agent: Agent):
        self.agents.append(agent)
        agent.runner = self
        self._agent_name_to_agent_map[agent.name] = Agent
        return agent

    def run(self):
        assert not self._closed, "You cannot run this runner again. This runner is closed. Use a new one"
        c = self.context
        c.summaries = Summaries(c.session)
        c.session.run(tf.global_variables_initializer())
        [agent.start() for agent in self.agents]

        for c.episode_id in range(c.n_episodes):
            c.frame_done = False
            self.env.stats_recorder.type = 'e' if c.should_eval_episode() else 't'
            c.frame_obs = self.env.reset()
            [agent.pre_episode() for agent in self.agents]
            while not c.frame_done:
                c.frame_id = c.total_steps
                [agent.pre_act() for agent in self.agents]
                actions = [agent.act() for agent in self.agents]
                actions = list(filter(lambda x: x is not None, actions))
                if len(actions) == 0:
                    logger.error(
                        "No agent returned any action! The environment cannot be stepped")
                    raise RuntimeError(
                        "No agent returned any action! The environment cannot be stepped")
                c.frame_action = actions[-1]
                c.frame_obs_next, c.frame_reward, c.frame_done, c.frame_info = self.env.step(c.frame_action)
                [agent.post_act() for agent in self.agents]
                c.frame_obs = c.frame_obs_next
            c.log_stats(average_over=100, end='\n' if c.episode_id % c.video_interval == 0 else '\r')
            [agent.post_episode() for agent in self.agents]

        logger.log('-------------------Done--------------------')
        c.log_stats(average_over=c.n_episodes)
        self._close()

    def _close(self):
        [agent.close() for agent in self.agents]
        for agent in self.agents:
            agent.runner = None
        self.agents.clear()
        self._agent_name_to_agent_map.clear()
        self._closed = True

    def get_agent(self, name):
        '''returns none if no agent found by this name'''
        return self._agent_name_to_agent_map.get(name)

    def get_agent_by_type(self, typ):
        '''returns list of agents which are instances of subclass of typ'''
        return list(filter(lambda agent: isinstance(agent, typ), self.agents))
