import gym

import RL
from RL.agents import (BasicStatsRecordingAgent, DDPGActAgent,  # noqa: F401
                       DDPGTrainAgent, EnvRenderingAgent,
                       ExperienceBufferAgent, ForceExploitControlAgent,
                       MatplotlibPlotAgent, ModelLoaderSaverAgent,
                       ParamNoiseAgent, ParamsCopyAgent, PygletLoopAgent,
                       RandomPlayAgent, RewardScalingAgent, SeedingAgent,
                       StatsLoggingAgent, TensorboardAgent, TensorFlowAgent)
from RL.common.atari_wrappers import wrap_atari
from RL.common.utils import need_conv_net
from RL.contexts import DDPGContext
from RL.models.ddpg_model import DDPGModel

c = DDPGContext()


def make(id):
    env = gym.make(id)  # type: gym.Env
    if need_conv_net(env.observation_space):
        env = wrap_atari(env, episode_life=c.atari_episode_life, clip_rewards=c.atari_clip_rewards, framestack_k=c.atari_framestack_k, frameskip_k=c.atari_frameskip_k, noop_max=c.atari_noop_max)
    return env


c.set_envs([make(c.env_id) for i in range(c.num_envs_to_make)])

r = RL.Runner(c, "runner")

# basics:
r.register_agent(TensorFlowAgent(c, "TensorFlowAgent"))
r.register_agent(SeedingAgent(c, "SeedingAgent"))
r.register_agent(RewardScalingAgent(c, "RewardScalingAgent"))

# core algo
r.register_agent(ForceExploitControlAgent(c, "ExploitControlAgent"))
r.register_agent(RandomPlayAgent(c, "MinimumExperienceAgent", play_for_steps=c.minimum_experience))
ddpg_act_agent = r.register_agent(DDPGActAgent(c, "DDPGActAgent"))
r.register_agent(ModelLoaderSaverAgent(c, "LoaderSaverAgent", ddpg_act_agent.model.params))
if not c.eval_mode:
    r.register_agent(ParamNoiseAgent(c, "ParamNoiseAgent", DDPGModel(c, "ParamNoiseAgent/noisy_model", num_critics=c.num_critics), ddpg_act_agent.model))
    exp_buff_agent = r.register_agent(ExperienceBufferAgent(c, "ExperienceBufferAgent"))
    ddpg_train_agent = r.register_agent(DDPGTrainAgent(c, "DDPGTrainAgent", ddpg_act_agent, exp_buff_agent))
    r.register_agent(ParamsCopyAgent(c, "TargetNetUpdateAgent", ddpg_act_agent.model.params, ddpg_train_agent.target_model.params, c.target_network_update_every, c.target_network_update_tau))

# rendering and visualizations:
if c.render:
    r.register_agent(EnvRenderingAgent(c, "RenderingAgent"))
r.register_agent(PygletLoopAgent(c, "PygletLoopAgent"))

# stats and graphs:
r.register_agent(BasicStatsRecordingAgent(c, "StatsRecordingAgent", frameskip=c.atari_frameskip_k if need_conv_net(c.envs[0].observation_space) else 1))
for env_id_no in range(c.num_envs):
    keys = list(filter(lambda k: k.startswith('Env-' + str(env_id_no)), RL.stats.stats_dict.keys()))
    r.register_agent(StatsLoggingAgent(c, "Env-{0}-StatsLoggingAgent".format(env_id_no), keys))
    r.register_agent(TensorboardAgent(c, "Env-{0}-TensorboardAgent".format(env_id_no), keys, 'Env-{0} Total Frames'.format(env_id_no)))
r.register_agent(MatplotlibPlotAgent(c, 'RPE', [(RL.stats.get('Env-0 Episode ID'), RL.stats.get('Env-0 Episode Reward'))], ['b-'], xlabel='Episode ID', ylabel='Reward', legend='RPE', auto_save=True, smoothing=c.matplotlib_smoothing))

# algo specific stats and graphs:
misc_keys = ['Critic Loss', 'Actor Loss', 'Total Updates', "Average Actor Critic Q", 'Exploration Divergence', 'Exploration Sigma']
r.register_agent(StatsLoggingAgent(c, 'Misc-StatsLoggingAgent', misc_keys))
r.register_agent(TensorboardAgent(c, 'Misc-TensorboardAgent', misc_keys, 'Env-0 Total Frames', log_every_episode=-1, log_every_step=100))

r.run()
