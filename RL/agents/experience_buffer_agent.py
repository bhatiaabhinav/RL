import RL
from RL.common.experience_buffer import ExperienceBuffer, Experience
from typing import List
import numpy as np


class ExperienceBufferAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, nsteps=None, buffer_length=None, buffer_size_MB=None):
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

    def add_to_experience_buffer(self, exp: Experience, env_id_no):
        nstep_buffer = self.nstep_buffers[env_id_no]
        reward_to_prop_back = exp.reward
        for old_exp in reversed(nstep_buffer):
            if old_exp.done:
                break
            # gamma is taken as an array in case the exp is multi stream and gamma is a vector
            reward_to_prop_back = np.asarray(self.context.gamma) * reward_to_prop_back
            old_exp.reward += reward_to_prop_back
            old_exp.next_state = exp.next_state
            old_exp.done = exp.done
        nstep_buffer.append(exp)
        if len(nstep_buffer) >= self.nsteps:
            self.experience_buffer.add(nstep_buffer.pop(0))
            assert len(nstep_buffer) == self.context.nsteps - 1

    def create_experiences(self):
        exps = []
        for i in range(self.context.num_envs):
            exps.append(Experience(self.runner.prev_obss[i], self.runner.actions[i], self.runner.rewards[i], self.runner.dones[i], self.runner.infos[i], self.runner.obss[i]))
        return exps

    def post_act(self):
        super().post_act()
        exps = self.create_experiences()
        for env_id_no, exp in enumerate(exps):
            self.add_to_experience_buffer(exp, env_id_no)
