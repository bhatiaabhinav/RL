import RL
import os
from RL.common.wrappers import MaxEpisodeStepsWrapper
from RL.agents import SeedingAgent, DDPGActAgent, ParamNoiseAgent, ExperienceBufferAgent, DDPGTrainAgent, ParamsCopyAgent, ModelLoaderSaverAgent, EnvRenderingAgent, DQNSensitivityVisualizerAgent, PygletLoopAgent  # noqa: F401
from RL.models.ddpg_model import DDPGModel


class MyContext(RL.Context):
    def wrappers(self, env):
        if 'Lunar' in self.env_id:
            env = MaxEpisodeStepsWrapper(env, 600)
        return env


c = MyContext()
r = RL.Runner(c)

r.register_agent(SeedingAgent(c, "SeedingAgent"))
ddpg_act_agent = r.register_agent(DDPGActAgent(c, "DDPGActAgent"))
r.register_agent(ModelLoaderSaverAgent(c, "LoaderSaverAgent", ddpg_act_agent.model.params, os.path.join(RL.logger.get_dir(), 'models'), c.load_model_dir, filename='model'))
if not c.eval_mode:
    r.register_agent(ParamNoiseAgent(c, "ParamNoiseAgent", DDPGModel, ddpg_act_agent.model, target_divergence=0.1))
    exp_buff_agent = r.register_agent(ExperienceBufferAgent(c, "ExperienceBufferAgent", nsteps=c.nsteps, buffer_length=c.experience_buffer_length, buffer_size_MB=c.experience_buffer_megabytes))
    ddpg_train_agent = r.register_agent(DDPGTrainAgent(c, "DDPGTrainAgent", ddpg_act_agent, exp_buff_agent))
    r.register_agent(ParamsCopyAgent(c, "DQNTargetNetUpdateAgent", ddpg_act_agent.model.params, ddpg_train_agent.target_model.params, c.target_network_update_every, c.target_network_update_tau))
if c.render:
    r.register_agent(EnvRenderingAgent(c, "RenderingAgent", auto_dispatch_on_render=False, episode_interval=c.render_interval))
# r.register_agent(PygletLoopAgent(c, "PygletLoopAgent"))

r.run()
