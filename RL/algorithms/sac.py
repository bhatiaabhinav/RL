import gym

import RL
from RL.agents import (BasicStatsRecordingAgent, SACActAgent,  # noqa: F401
                       SACTrainAgent, EnvRenderingAgent,
                       ExperienceBufferAgent, ModelLoaderSaverAgent,
                       ParamsCopyAgent, PygletLoopAgent,
                       SeedingAgent, StatsLoggingAgent, TensorboardAgent,
                       TensorFlowAgent)
from RL.common.atari_wrappers import wrap_atari
from RL.common.utils import need_conv_net
from RL.contexts import SACContext
import RL.envs


c = SACContext()


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

# core algo
sac_act_agent = r.register_agent(SACActAgent(c, "SACActAgent"))
r.register_agent(ModelLoaderSaverAgent(c, "LoaderSaverAgent", sac_act_agent.model.get_vars()))
if not c.eval_mode:
    exp_buff_agent = r.register_agent(ExperienceBufferAgent(c, "ExperienceBufferAgent"))
    sac_train_agent = r.register_agent(SACTrainAgent(c, "SACTrainAgent", sac_act_agent, exp_buff_agent))
    r.register_agent(ParamsCopyAgent(c, "TargetNetUpdateAgent", sac_act_agent.model.get_vars('valuefn0', 'running_stats'), sac_train_agent.target_model.get_vars('valuefn0', 'running_stats'), c.target_network_update_every, c.target_network_update_tau))

# rendering and visualizations:
if c.render:
    r.register_agent(EnvRenderingAgent(c, "RenderingAgent", auto_dispatch_on_render=False))
r.register_agent(PygletLoopAgent(c, "PygletLoopAgent"))

# stats and graphs:
r.register_agent(BasicStatsRecordingAgent(c, "StatsRecordingAgent", frameskip=c.atari_frameskip_k if need_conv_net(c.envs[0].observation_space) else 1))
for env_id_no in range(c.num_envs):
    keys = list(filter(lambda k: k.startswith('Env-' + str(env_id_no)), RL.stats.stats_dict.keys()))
    r.register_agent(StatsLoggingAgent(c, "Env-{0}-StatsLoggingAgent".format(env_id_no), keys))
    r.register_agent(TensorboardAgent(c, "Env-{0}-TensorboardAgent".format(env_id_no), keys, 'Env-{0} Total Frames'.format(env_id_no)))
# algo specific stats and graphs:
misc_keys = ['ValueFn Loss', 'Critic Loss', 'Actor Loss', 'Total Updates', "Average Actor Critic Q", "Average Action LogStd", "Average Action LogPi"]
r.register_agent(StatsLoggingAgent(c, 'Misc-StatsLoggingAgent', misc_keys))
r.register_agent(TensorboardAgent(c, 'Misc-TensorboardAgent', misc_keys, 'Env-0 Total Frames', log_every_episode=-1, log_every_step=100))

r.run()
