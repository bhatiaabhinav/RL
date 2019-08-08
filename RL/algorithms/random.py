import RL
from RL.agents import SeedingAgent, RandomPlayAgent, EnvRenderingAgent, PygletLoopAgent, BasicStatsRecordingAgent, StatsLoggingAgent, TensorFlowAgent, TensorboardAgent, ForceExploitControlAgent, RewardScalingAgent
import gym

c = RL.Context()
c.set_envs([gym.make(c.env_id) for i in range(c.num_envs_to_make)])
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

# stats and graphs:
r.register_agent(BasicStatsRecordingAgent(c, "StatsRecordingAgent"))
for env_id_no in range(c.num_envs):
    keys = list(filter(lambda k: k.startswith('Env-' + str(env_id_no)), RL.stats.stats_dict.keys()))
    r.register_agent(StatsLoggingAgent(c, "Env-{0}-StatsLoggingAgent".format(env_id_no), keys))
    r.register_agent(TensorboardAgent(c, "Env-{0}-TensorboardAgent".format(env_id_no), keys, 'Env-{0} Total Frames'.format(env_id_no)))

r.run()
