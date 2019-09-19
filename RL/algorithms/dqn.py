import gym

import RL
from RL.agents import (BasicStatsRecordingAgent, DQNActAgent,  # noqa: F401
                       DQNSensitivityVisualizerAgent, DQNTrainAgent,
                       EnvRenderingAgent, ExperienceBufferAgent,
                       ForceExploitControlAgent, LinearAnnealingAgent,
                       MatplotlibPlotAgent, ModelLoaderSaverAgent,
                       ParamsCopyAgent, PygletLoopAgent, RandomPlayAgent,
                       RewardScalingAgent, SeedingAgent, StatsLoggingAgent,
                       TensorboardAgent, TensorFlowAgent)
from RL.common.atari_wrappers import (ClipRewardEnv, EpisodicLifeEnv,
                                      FireResetEnv, NoopResetEnv, wrap_atari)
from RL.common.utils import need_conv_net
from RL.common.wrappers import FrameSkipWrapper
from RL.contexts import DQNContext

c = DQNContext()


def make(id):
    env = gym.make(id)  # type: gym.Env
    if need_conv_net(env.observation_space):
        env = wrap_atari(env, episode_life=c.atari_episode_life, clip_rewards=c.atari_clip_rewards, framestack_k=c.atari_framestack_k, frameskip_k=c.atari_frameskip_k, noop_max=c.atari_noop_max)
    if 'ram' in id and 'v4' in id:  # for playing atari from ram
        if c.atari_episode_life:
            env = EpisodicLifeEnv(env)
        env = NoopResetEnv(env, noop_max=30)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = FrameSkipWrapper(env, skip=c.atari_frameskip_k)
        if c.atari_clip_rewards:
            env = ClipRewardEnv(env)
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
dqn_act_agent = r.register_agent(DQNActAgent(c, "DQNActAgent"))
r.register_agent(ModelLoaderSaverAgent(c, "LoaderSaverAgent", dqn_act_agent.model.params))
if not c.eval_mode:
    r.register_agent(LinearAnnealingAgent(c, "EpsilonAnnealingAgent", 'epsilon', c.minimum_experience, c.epsilon, c.final_epsilon, c.epsilon_anneal_over))
    exp_buff_agent = r.register_agent(ExperienceBufferAgent(c, "ExperienceBufferAgent"))
    dqn_train_agent = r.register_agent(DQNTrainAgent(c, "DQNTrainAgent", dqn_act_agent, exp_buff_agent))
    r.register_agent(ParamsCopyAgent(c, "DQNTargetNetUpdateAgent", dqn_act_agent.model.params, dqn_train_agent.target_model.params, c.target_network_update_every, c.target_network_update_tau))

# rendering and visualizations:
if c.sensitivity_visualizer:
    r.register_agent(DQNSensitivityVisualizerAgent(c, "DQNSensitivityVisualizerAgent", dqn_act_agent.model))
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

misc_keys = ['epsilon', 'Q_loss:DQNActAgent/main_brain/default/Q_loss', 'Q_mpe:DQNActAgent/main_brain/default/Q_mpe', 'mb_av_V:DQNActAgent/main_brain/default/mb_av_V', 'Q Updates']
r.register_agent(StatsLoggingAgent(c, 'Misc-StatsLoggingAgent', misc_keys))
r.register_agent(TensorboardAgent(c, 'Misc-TensorboardAgent', misc_keys, 'Env-0 Total Frames', log_every_episode=-1, log_every_step=100))

r.run()
