import gym

import RL
from RL.agents import (BasicStatsRecordingAgent, EnvRenderingAgent,
                       ForceExploitControlAgent, MatplotlibPlotAgent,
                       PygletLoopAgent, RandomPlayAgent, RewardScalingAgent,
                       SeedingAgent, StatsLoggingAgent, TensorboardAgent,
                       TensorFlowAgent)
from RL.common.wrappers import wrap_standard


c = RL.Context()
c.set_env(wrap_standard(gym.make(c.env_id), c))
r = RL.Runner(c, "runner")

# basics:
r.register_agent(TensorFlowAgent(c, "TensorFlowAgent"))
r.register_agent(SeedingAgent(c, "SeedingAgent"))
r.register_agent(RewardScalingAgent(c, "RewardScalingAgent"))

# core algo:
r.register_agent(ForceExploitControlAgent(c, "ExploitControlAgent"))
r.register_agent(RandomPlayAgent(c, "RandomPlayAgent"))

# rendering:
if c.render:
    r.register_agent(EnvRenderingAgent(c, "RenderingAgent"))
r.register_agent(PygletLoopAgent(c, "PygletLoopAgent"))

# stats record:
r.register_agent(BasicStatsRecordingAgent(c, "StatsRecordingAgent"))
# stats log:
keys = list(filter(lambda k: k.startswith('Env-0'), RL.stats.stats_dict.keys()))
r.register_agent(StatsLoggingAgent(c, "Env-0-StatsLoggingAgent", keys, poll_every_episode=1))
# stats plot:
r.register_agent(TensorboardAgent(c, "Env-0-TensorboardAgent", keys, 'Env-0 Total Frames'))
r.register_agent(MatplotlibPlotAgent(c, 'RPE', [(RL.stats.get('Env-0 Episode ID'), RL.stats.get('Env-0 Episode Reward'))], ['b-'], xlabel='Episode ID', ylabel='Reward', legend='RPE', auto_save=True, smoothing=c.matplotlib_smoothing))

r.run()


"""
python -m RL.algorithms.random --env_id=BreakoutNoFrameskip-v4 --experiment_name=Random --atari_clip_rewards=False --atari_episode_life=True --render=True
"""
