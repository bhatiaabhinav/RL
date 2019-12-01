import gym
import safety_gym  # noqa

import RL
import RL.envs
from RL.agents import BasicStatsRecordingAgent  # noqa: F401
from RL.agents import (EnvRenderingAgent, ExperienceBufferAgent,  # noqa
                       ForceExploitControlAgent, MatplotlibPlotAgent,
                       ModelLoaderSaverAgent, ParamsCopyAgent, PygletLoopAgent,
                       RandomPlayAgent, RewardScalingAgent, SACActAgent,
                       SACTrainAgent, SeedingAgent, StatsLoggingAgent,
                       TensorboardAgent, TensorFlowAgent)
from RL.common.wrappers import wrap_standard
from RL.contexts import SACContext

c = SACContext()
c.set_env(wrap_standard(gym.make(c.env_id), c))

r = RL.Runner(c, "runner")

# basics:
r.register_agent(TensorFlowAgent(c, "TensorFlowAgent"))
r.register_agent(SeedingAgent(c, "SeedingAgent"))
r.register_agent(RewardScalingAgent(c, "RewardScalingAgent"))

# core algo
r.register_agent(ForceExploitControlAgent(c, "ExploitControlAgent"))
r.register_agent(RandomPlayAgent(c, "MinimumExperienceAgent", play_for_steps=c.minimum_experience))
sac_act_agent = r.register_agent(SACActAgent(c, "SACActAgent"))
r.register_agent(ModelLoaderSaverAgent(c, "LoaderSaverAgent", sac_act_agent.model.get_vars()))
if not c.eval_mode:
    exp_buff_agent = r.register_agent(ExperienceBufferAgent(c, "ExperienceBufferAgent"))
    sac_train_agent = r.register_agent(SACTrainAgent(c, "SACTrainAgent", sac_act_agent, exp_buff_agent))
    r.register_agent(ParamsCopyAgent(c, "TargetNetUpdateAgent", sac_act_agent.model.get_vars('valuefn0', 'running_stats'), sac_train_agent.target_model.get_vars('valuefn0', 'running_stats'), c.target_network_update_every, c.target_network_update_tau))

# rendering and visualizations:
if c.render:
    r.register_agent(EnvRenderingAgent(c, "RenderingAgent"))
r.register_agent(PygletLoopAgent(c, "PygletLoopAgent"))

# stats record:
r.register_agent(BasicStatsRecordingAgent(c, "StatsRecordingAgent"))

# stats log:
keys = list(filter(lambda k: k.startswith('Env-0'), RL.stats.stats_dict.keys()))
misc_keys = ['ValueFn Loss', 'Critic Loss', 'Actor Loss', 'Total Updates', "Average Actor Critic Q", "Average Action LogStd", "Average Action LogPi"]
r.register_agent(StatsLoggingAgent(c, "Env-0-StatsLoggingAgent", keys + misc_keys, poll_every_episode=1))

# stats plot:
r.register_agent(TensorboardAgent(c, "Env-0-TensorboardAgent", keys, 'Env-0 Total Frames'))
r.register_agent(TensorboardAgent(c, 'Misc-TensorboardAgent', misc_keys, 'Env-0 Total Frames', log_every_episode=-1, log_every_step=100))
# r.register_agent(MatplotlibPlotAgent(c, 'RPE', [(RL.stats.get('Env-0 Episode ID'), RL.stats.get('Env-0 Episode Reward'))], ['b-'], xlabel='Episode ID', ylabel='Reward', legend='RPE', auto_save=True, smoothing=c.matplotlib_smoothing))

r.run()


"""
python -m RL.algorithms.sac --env_id=HalfCheetah-v2 --experiment_name=sac --num_steps_to_run=500000 --normalize_observations=False --alpha=0.2 --actor_learning_rate=0.0003 --learning_rate=0.0003 --target_network_update_tau=0.005 --exploit_every=8 --minimum_experience=10000 --logstd_min=-20 --logstd_max=2 --num_critics=2 --init_scale=None --l2_reg=0 --train_every=1 --experience_buffer_length=1000000 --minibatch_size=100 --hidden_layers=[256,256] --gamma=0.99 --layer_norm=False --clip_gradients=None --record_returns=False --reward_scaling=1 --ignore_done_on_timelimit=True
"""

# Keeping ignore_done_on_timelimit=False for Safexp- environments because:
#   - They have time as part of state space
#   - It was anyway not going to work. Because the num_steps info is contained in env.config disctionary instead of env.spec.max_episode_steps
"""
 python -m RL.algorithms.sac --env_id=Safexp-PointGoal0-v0 --experiment_name=sac_ln --num_steps_to_run=500000 --normalize_observations=False --alpha=0.2 --actor_learning_rate=0.0003 --learning_rate=0.0003 --target_network_update_tau=0.005 --exploit_every=8 --minimum_experience=10000 --logstd_min=-20 --logstd_max=2 --num_critics=2 --init_scale=None --l2_reg=0 --train_every=1 --experience_buffer_length=1000000 --minibatch_size=100 --hidden_layers=[256,256] --gamma=0.99 --layer_norm=True --clip_gradients=None --record_returns=False --reward_scaling=1 --ignore_done_on_timelimit=False
"""

"""
python -m RL.algorithms.sac --env_id=Safexp-PointGoal0-v0 --experiment_name=sac --num_steps_to_run=10000000 --normalize_observations=False --alpha=0.2 --actor_learning_rate=0.0001 --learning_rate=0.001 --target_network_update_tau=0.005 --exploit_every=8 --minimum_experience=10000 --logstd_min=-20 --logstd_max=2 --num_critics=2 --init_scale=None --l2_reg=0.01 --train_every=1 --experience_buffer_length=1000000 --minibatch_size=100 --hidden_layers=[256,256] --gamma=0.99 --layer_norm=False --clip_gradients=None --record_returns=False --reward_scaling=1 --ignore_done_on_timelimit=False
"""

# this one worked:
"""
python -m RL.algorithms.sac --env_id=Safexp-PointGoal1-v0 --experiment_name=sac_ln --num_steps_to_run=10000000 --normalize_observations=False --alpha=0.2 --actor_learning_rate=0.0001 --learning_rate=0.001 --target_network_update_tau=0.005 --exploit_every=8 --minimum_experience=10000 --logstd_min=-20 --logstd_max=2 --num_critics=2 --init_scale=None --l2_reg=0 --train_every=1 --experience_buffer_length=1000000 --minibatch_size=100 --hidden_layers=[256,256] --gamma=0.99 --layer_norm=True --clip_gradients=None --record_returns=False --reward_scaling=100 --ignore_done_on_timelimit=False
"""
