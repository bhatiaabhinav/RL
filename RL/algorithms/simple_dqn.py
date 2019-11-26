import gym

import RL
from RL.agents import (BasicStatsRecordingAgent, SimpleDQNActAgent,  # noqa: F401
                       DQNSensitivityVisualizerAgent, SimpleDQNTrainAgent,
                       EnvRenderingAgent, ExperienceBufferAgent,
                       ForceExploitControlAgent, LinearAnnealingAgent,
                       MatplotlibPlotAgent, ModelLoaderSaverAgent,
                       ParamsCopyAgent, PygletLoopAgent, RandomPlayAgent,
                       RewardScalingAgent, SeedingAgent, StatsLoggingAgent,
                       TensorboardAgent, TensorFlowAgent)
from RL.common.atari_wrappers import (ClipRewardEnv, EpisodicLifeEnv,
                                      FireResetEnv, NoopResetEnv)
from RL.common.utils import need_conv_net
from RL.common.wrappers import FrameSkipWrapper
from gym.wrappers import AtariPreprocessing, FrameStack
from RL.contexts import DQNContext

c = DQNContext()


def make(id):
    env = gym.make(id)  # type: gym.Env
    if need_conv_net(env.observation_space):
        env = FireResetEnv(env)
        env = AtariPreprocessing(env, c.atari_noop_max, c.atari_frameskip_k, terminal_on_life_loss=c.atari_episode_life)
        env = FrameStack(env, c.atari_framestack_k)
        c.frameskip = c.atari_frameskip_k
        # print(env)
    elif '-ram' in id:  # for playing atari from ram
        if c.atari_episode_life:
            env = EpisodicLifeEnv(env)
        env = NoopResetEnv(env, noop_max=c.atari_noop_max)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
            RL.logger.log("Fire reset being used")
        env = FrameSkipWrapper(env, skip=c.atari_frameskip_k)
        # if c.atari_clip_rewards:
        #     env = ClipRewardEnv(env)
        c.frameskip = c.atari_frameskip_k
        print(env)
    else:
        if c.frameskip > 1:
            env = FrameSkipWrapper(env, skip=c.frameskip)
    return env


c.set_env(make(c.env_id))

r = RL.Runner(c, "runner")

# basics:
r.register_agent(TensorFlowAgent(c, "TensorFlowAgent"))
r.register_agent(SeedingAgent(c, "SeedingAgent"))
r.register_agent(RewardScalingAgent(c, "RewardScalingAgent"))

# core algo
r.register_agent(ForceExploitControlAgent(c, "ExploitControlAgent"))
r.register_agent(RandomPlayAgent(c, "MinimumExperienceAgent", play_for_steps=c.minimum_experience))
dqn_act_agent = r.register_agent(SimpleDQNActAgent(c, "DQNActAgent"))
r.register_agent(ModelLoaderSaverAgent(c, "LoaderSaverAgent", dqn_act_agent.model.params))
if not c.eval_mode:
    r.register_agent(LinearAnnealingAgent(c, "EpsilonAnnealingAgent", 'epsilon', c.minimum_experience, c.epsilon, c.final_epsilon, c.epsilon_anneal_over))
    exp_buff_agent = r.register_agent(ExperienceBufferAgent(c, "ExperienceBufferAgent"))
    dqn_train_agent = r.register_agent(SimpleDQNTrainAgent(c, "DQNTrainAgent", dqn_act_agent, exp_buff_agent))
    r.register_agent(ParamsCopyAgent(c, "DQNTargetNetUpdateAgent", dqn_act_agent.model.params, dqn_train_agent.target_model.params, c.target_network_update_every, c.target_network_update_tau))

# rendering and visualizations:
if c.sensitivity_visualizer:
    r.register_agent(DQNSensitivityVisualizerAgent(c, "DQNSensitivityVisualizerAgent", dqn_act_agent.model))
if c.render:
    r.register_agent(EnvRenderingAgent(c, "RenderingAgent"))
r.register_agent(PygletLoopAgent(c, "PygletLoopAgent"))

# stats record:
r.register_agent(BasicStatsRecordingAgent(c, "StatsRecordingAgent"))
# stats log:
keys = list(filter(lambda k: k.startswith('Env-0'), RL.stats.stats_dict.keys()))
misc_keys = ['epsilon', 'Q_loss:DQNActAgent/main_brain/Q_loss', 'Q_mpe:DQNActAgent/main_brain/Q_mpe', 'mb_av_V:DQNActAgent/main_brain/mb_av_V', 'Q Updates', 'att_ent']
r.register_agent(StatsLoggingAgent(c, "Env-0-StatsLoggingAgent", keys + misc_keys, poll_every_episode=1))
# stats plot:
r.register_agent(TensorboardAgent(c, "Env-0-TensorboardAgent", keys, 'Env-0 Total Frames'))
r.register_agent(TensorboardAgent(c, 'Misc-TensorboardAgent', misc_keys + ['att_hist'], 'Env-0 Total Frames', log_every_episode=-1, log_every_step=100))
# r.register_agent(MatplotlibPlotAgent(c, 'RPE', [(RL.stats.get('Env-0 Episode ID'), RL.stats.get('Env-0 Episode Reward'))], ['b-'], xlabel='Episode ID', ylabel='Reward', legend='RPE', auto_save=True, smoothing=c.matplotlib_smoothing))

r.run()
