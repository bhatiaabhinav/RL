import RL
from RL.common.experience_buffer import ExperienceBuffer, Experience
from typing import List
import numpy as np


class ExperienceBufferAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, nsteps=None, buffer_length=None, buffer_size_MB=None, ignore_done_on_timelimit=None):
        super().__init__(context, name)
        self.nsteps = nsteps
        self.buffer_length = buffer_length
        self.buffer_size_MB = buffer_size_MB
        if self.nsteps is None:
            self.nsteps = self.context.nsteps
        if self.buffer_length is None:
            self.buffer_length = self.context.experience_buffer_length
        if self.buffer_size_MB is None:
            self.buffer_size_MB = self.context.experience_buffer_megabytes
        if self.buffer_length is None:
            self.experience_buffer = ExperienceBuffer(size_in_bytes=self.buffer_size_MB * (1024**2))
        else:
            self.experience_buffer = ExperienceBuffer(length=self.buffer_length)
        self.nstep_buffers = [[]] * self.context.num_envs  # type: List[Experience]
        if ignore_done_on_timelimit is None:
            ignore_done_on_timelimit = self.context.ignore_done_on_timelimit
        if hasattr(self.context.env.spec, 'max_episode_steps') and self.context.env.spec.max_episode_steps and ignore_done_on_timelimit:
            RL.logger.info("{0}: This is a timelimit environment. Done signal will be ignored if either TimeLimit.truncated info is provided or if not provided, then if length of episode is same as env.spec.max_episode_steps. Set ignore_done_on_timelimit to false to disable this behavior.".format(self.name))
            self.ignore_done_on_timelimit = True
        else:
            self.ignore_done_on_timelimit = False

    def add_to_experience_buffer(self, exp: Experience, env_id_no):
        nstep_buffer = self.nstep_buffers[env_id_no]
        reward_to_prop_back = exp.reward
        cost_to_prop_back = exp.cost
        for old_exp in reversed(nstep_buffer):
            if old_exp.done:
                break
            # gamma is taken as an array in case the exp is multi stream and gamma is a vector
            # if not np.asarray(self.context.gamma).shape == np.asarray(reward_to_prop_back).shape:
                # RL.logger.error("Number of gammas should be same as number of rewards")
                # raise Exception("Number of gammas should be same as number of rewards")
            reward_to_prop_back = self.context.gamma * reward_to_prop_back
            cost_to_prop_back = self.context.cost_gamma * cost_to_prop_back
            old_exp.reward += reward_to_prop_back
            old_exp.cost += cost_to_prop_back
            old_exp.next_state = exp.next_state
            old_exp.done = exp.done
        nstep_buffer.append(exp)
        if len(nstep_buffer) >= self.nsteps:
            self.experience_buffer.add(nstep_buffer.pop(0))
            assert len(nstep_buffer) == self.context.nsteps - 1

    def get_state(self, env_id_no):
        return self.runner.prev_obss[env_id_no]

    def get_action(self, env_id_no):
        return self.runner.actions[env_id_no]

    def get_reward(self, env_id_no):
        return self.runner.rewards[env_id_no]

    def get_cost(self, env_id_no):
        return self.runner.costs[env_id_no]

    def get_done(self, env_id_no):
        done = self.runner.dones[env_id_no]
        if done:
            assert int(done) == 1, "done was True but not 1"
        if done and self.ignore_done_on_timelimit:
            info = self.get_info(env_id_no)
            if 'TimeLimit.truncated' in info:
                done = not info.get('TimeLimit.truncated')
            elif self.runner.episode_step_ids[env_id_no] + 1 == self.context.envs[env_id_no].spec.max_episode_steps:
                done = False
        return done

    def get_info(self, env_id_no):
        return self.runner.infos[env_id_no]

    def get_next_state(self, env_id_no):
        return self.runner.obss[env_id_no]

    def create_experiences(self):
        exps = []
        for i in range(self.context.num_envs):
            exps.append(Experience(self.runner.prev_obss[i], self.runner.actions[i], self.runner.rewards[i], self.runner.dones[i], self.runner.infos[i], self.runner.obss[i]))
        return exps

    def post_act(self):
        super().post_act()
        for env_id_no in range(self.context.num_envs):
            s, a, r, d, i, ns, c = self.get_state(env_id_no), self.get_action(env_id_no), self.get_reward(env_id_no), self.get_done(env_id_no), self.get_info(env_id_no), self.get_next_state(env_id_no), self.get_cost(env_id_no)
            exp = Experience(s, a, r, d, i, ns, cost=c)
            self.add_to_experience_buffer(exp, env_id_no)
