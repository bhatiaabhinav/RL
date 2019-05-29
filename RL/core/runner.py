from . import logger
from .context import Context
from .agent import Agent
from typing import List
import numpy as np


class Runner:
    '''runs the registered agents on the gym environments in the context. The env stepping logic is synchronous. And an agent is provided feedback from all envs simulatenously and must provide actions for all envs'''
    def __init__(self, context: Context, name, num_steps_to_run=None, num_episodes_to_run=None):
        self.name = name
        self.context = context
        self.agents = []  # type: List[Agent]
        self.envs = self.context.envs
        self.num_steps_to_run = num_steps_to_run
        self.num_episodes_to_run = num_episodes_to_run
        if self.num_steps_to_run is None:
            self.num_steps_to_run = self.context.num_steps_to_run
        if self.num_episodes_to_run is None:
            self.num_episodes_to_run = self.context.num_episodes_to_run
        self.num_stepss = np.array([0] * self.num_envs)
        self.num_episode_stepss = np.array([0] * self.num_envs)
        self.num_episodess = np.array([0] * self.num_envs)
        self.prev_obss = [None] * self.num_envs
        self.obss = [None] * self.num_envs
        self.rewards = np.zeros(self.num_envs)
        self.dones = np.array([True] * self.num_envs)
        self.infos = [{}] * self.num_envs
        self._agent_name_to_agent_map = {}

    @property
    def env(self):
        return self.envs[0]

    @property
    def num_envs(self):
        return self.context.num_envs

    @property
    def num_steps(self):
        return self.num_stepss[0]

    @property
    def num_episode_steps(self):
        return self.num_episode_stepss[0]

    @property
    def num_episodes(self):
        return self.num_episodess[0]

    @property
    def step_ids(self):
        return self.num_stepss

    @property
    def episode_step_ids(self):
        return self.num_episode_stepss

    @property
    def episode_ids(self):
        return self.num_episodess

    @property
    def step_id(self):
        return self.step_ids[0]

    @property
    def episode_step_id(self):
        return self.episode_step_ids[0]

    @property
    def episode_id(self):
        return self.episode_ids[0]

    @property
    def obs(self):
        return self.obss[0]

    @property
    def action(self):
        return self.actions[0]

    @property
    def reward(self):
        return self.rewards[0]

    @property
    def done(self):
        return self.dones[0]

    @property
    def info(self):
        return self.infos[0]

    @property
    def prev_obs(self):
        return self.prev_obss[0]

    def register_agent(self, agent: Agent):
        self.agents.append(agent)
        agent.runner = self
        self._agent_name_to_agent_map[agent.name] = Agent
        return agent

    def enabled_agents(self):
        return list(filter(lambda agent: agent.enabled, self.agents))

    def should_stop(self):
        return np.min(self.num_stepss) >= self.num_steps_to_run or np.min(self.num_episodess) >= self.num_episodes_to_run

    def run(self):
        [agent.start() for agent in self.enabled_agents()]
        need_reset_env_id_nos = np.asarray(list(range(self.num_envs)))
        while not self.should_stop():
            # do the resets
            for env_id_no in need_reset_env_id_nos:
                self.prev_obss[env_id_no] = self.obss[env_id_no]
                self.obss[env_id_no] = self.envs[env_id_no].reset()
                self.rewards[env_id_no] = 0
                self.dones[env_id_no] = False
                self.infos[env_id_no] = {}
                self.num_episode_stepss[env_id_no] = 0
            # pre episode for envs which just got resetted
            if len(need_reset_env_id_nos) > 0:
                [agent.pre_episode(env_id_nos=need_reset_env_id_nos) for agent in self.enabled_agents()]
            # pre act
            [agent.pre_act() for agent in self.enabled_agents()]
            # act
            actions_per_agent = [agent.act() for agent in self.enabled_agents()]
            actions_per_agent = list(filter(lambda x: x is not None, actions_per_agent))
            # step
            self.prev_obss = [obs for obs in self.obss]
            self.actions = [None for env in self.envs]
            for actions in actions_per_agent:
                assert len(actions) == self.num_envs, "Please provide a list of actions for all envs. Can provide None action if do not want to provide for a particular env"
                for env_id_no, action in enumerate(actions):
                    self.actions[env_id_no] = action
            for env_id_no, (env, action) in enumerate(zip(self.envs, self.actions)):
                if action is None:
                    logger.error("No agent returned any action for env#{0} ! The environment cannot be stepped".format(env_id_no))
                    raise RuntimeError("No agent returned any action for env#{0} ! The environment cannot be stepped".format(env_id_no))
                self.obss[env_id_no], self.rewards[env_id_no], self.dones[env_id_no], self.infos[env_id_no] = env.step(action)
            # post act
            [agent.post_act() for agent in self.enabled_agents()]
            # post episode for envs which are done:
            need_reset_env_id_nos = np.asarray(list(filter(lambda i: self.dones[i], range(self.num_envs))))
            if len(need_reset_env_id_nos) > 0:
                [agent.post_episode(env_id_nos=need_reset_env_id_nos) for agent in self.enabled_agents()]
            # increment steps and episodes
            self.num_episode_stepss = self.num_episode_stepss + 1
            self.num_stepss = self.num_stepss + 1
            if len(need_reset_env_id_nos) > 0:
                self.num_episodess[need_reset_env_id_nos] = self.num_episodess[need_reset_env_id_nos] + 1
        logger.log('-------------------Done--------------------')
        [agent.close() for agent in self.agents]
        [agent.post_close() for agent in self.agents]
        [env.close() for env in self.envs]

        # for c.episode_id in range(c.n_episodes):
        #     c.frame_done = False
        #     self.env.stats_recorder.type = 'e' if c.should_eval_episode() else 't'
        #     c.frame_obs = self.env.reset()
        #     while not c.frame_done:
        #         c.frame_id = c.total_steps
        #         [agent.pre_act() for agent in self.enabled_agents()]
        #         actions = [agent.act() for agent in self.enabled_agents()]
        #         actions = list(filter(lambda x: x is not None, actions))
        #         if len(actions) == 0:
        #             logger.error(
        #                 "No agent returned any action! The environment cannot be stepped")
        #             raise RuntimeError(
        #                 "No agent returned any action! The environment cannot be stepped")
        #         c.frame_action = actions[-1]
        #         c.frame_obs_next, c.frame_reward, c.frame_done, c.frame_info = self.env.step(c.frame_action)
        #         [agent.post_act() for agent in self.enabled_agents()]
        #         c.frame_obs = c.frame_obs_next
        #     c.log_stats(average_over=100, end='\n' if c.episode_id % c.video_interval == 0 else '\r')
        #     [agent.post_episode() for agent in self.enabled_agents()]

        # logger.log('-------------------Done--------------------')
        # c.log_stats(average_over=c.n_episodes)
        # [agent.close() for agent in self.agents]
        # [env.close() for env in self.envs]

    def get_agent(self, name):
        '''returns none if no agent found by this name'''
        return self._agent_name_to_agent_map.get(name)

    def get_agent_by_type(self, typ):
        '''returns list of agents which are instances of subclass of typ'''
        return list(filter(lambda agent: isinstance(agent, typ), self.agents))
