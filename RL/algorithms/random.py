import RL
from RL.agents import SeedingAgent, RandomPlayAgent, EnvRenderingAgent, PygletLoopAgent, BasicStatsRecordingAgent, StatsLoggingAgent
import gym

c = RL.Context()
c.set_envs([gym.make(c.env_id) for i in range(1)])
r = RL.Runner(c, "runner", 1e10, num_episodes_to_run=c.n_episodes)

r.register_agent(SeedingAgent(c, "SeedingAgent", seed=c.seed))
r.register_agent(RandomPlayAgent(c, "RandomPlayAgent"))
r.register_agent(BasicStatsRecordingAgent(c, "StatsRecordingAgent"))
r.register_agent(StatsLoggingAgent(c, "StatsLoggingAgent", ['Ep Id', 'Ep R', 'Av Ep R (100)', 'Ep L', 'Total Steps', 'Ep Exploit Mode', 'Av Exploit Ep R (100)', 'Total Episodes'], poll_every_episode=1, keys_prefix='Env-0 '))
if c.render:
    r.register_agent(EnvRenderingAgent(c, "RenderingAgent", auto_dispatch_on_render=False, episode_interval=c.render_interval))
r.register_agent(PygletLoopAgent(c, "PygletLoopAgent"))

r.run()
