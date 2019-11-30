import gym
import safety_gym  # noqa

import RL
import RL.envs
from RL.agents import BasicStatsRecordingAgent
from RL.agents import (EnvRenderingAgent, ExperienceBufferAgent,  # noqa
                       ForceExploitControlAgent, MatplotlibPlotAgent,
                       ModelLoaderSaverAgent, ParamsCopyAgent, PygletLoopAgent,
                       RandomPlayAgent, RewardScalingAgent, SafeSACActAgent,
                       SafeSACTrainAgent, SeedingAgent, StatsLoggingAgent,
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
safe_sac_act_agent = r.register_agent(SafeSACActAgent(c, "SafeSACActAgent"))
r.register_agent(ModelLoaderSaverAgent(c, "LoaderSaverAgent", safe_sac_act_agent.model.get_vars()))
if not c.eval_mode:
    exp_buff_agent = r.register_agent(ExperienceBufferAgent(c, "ExperienceBufferAgent"))
    safe_sac_train_agent = r.register_agent(SafeSACTrainAgent(c, "SafeSACTrainAgent", safe_sac_act_agent, exp_buff_agent))
    r.register_agent(ParamsCopyAgent(c, "TargetNetUpdateAgent", safe_sac_act_agent.model.get_vars('valuefn0', 'valuefn1', 'running_stats'), safe_sac_train_agent.target_model.get_vars('valuefn0', 'valuefn1', 'running_stats'), c.target_network_update_every, c.target_network_update_tau))

# rendering and visualizations:
if c.render:
    r.register_agent(EnvRenderingAgent(c, "RenderingAgent"))
r.register_agent(PygletLoopAgent(c, "PygletLoopAgent"))

# stats record:
r.register_agent(BasicStatsRecordingAgent(c, "StatsRecordingAgent"))

# stats log:
keys = list(filter(lambda k: k.startswith('Env-0'), RL.stats.stats_dict.keys()))
misc_keys = ['ValueFn Loss', "Safety ValueFn Loss", 'Critic Loss', "Safety Critic Loss", 'Actor Loss', 'Total Updates', "Average Actor Critic Q", "Average Actor Critic Safety Q", "Average Action LogStd", "Average Action LogPi"]
r.register_agent(StatsLoggingAgent(c, "Env-0-StatsLoggingAgent", keys + misc_keys, poll_every_episode=1))

# stats plot:
r.register_agent(TensorboardAgent(c, "Env-0-TensorboardAgent", keys, 'Env-0 Total Frames'))
r.register_agent(TensorboardAgent(c, 'Misc-TensorboardAgent', misc_keys, 'Env-0 Total Frames', log_every_episode=-1, log_every_step=100))
# r.register_agent(MatplotlibPlotAgent(c, 'RPE', [(RL.stats.get('Env-0 Episode ID'), RL.stats.get('Env-0 Episode Reward'))], ['b-'], xlabel='Episode ID', ylabel='Reward', legend='RPE', auto_save=True, smoothing=c.matplotlib_smoothing))
# r.register_agent(MatplotlibPlotAgent(c, 'CPE', [(RL.stats.get('Env-0 Episode ID'), RL.stats.get('Env-0 Episode Cost'))], ['b-'], xlabel='Episode ID', ylabel='Cost', legend='CPE', auto_save=True, smoothing=c.matplotlib_smoothing))


r.run()


"""
python -m RL.algorithms.safe_sac --env_id=MyPointCircleFinite-v0 --experiment_name=safesac_ln_vs_cpo --num_steps_to_run=150000 --normalize_observations=False --alpha=0.2 --actor_learning_rate=0.0001 --learning_rate=0.001 --target_network_update_tau=0.005 --exploit_every=8 --minimum_experience=10000 --logstd_min=-20 --logstd_max=2 --num_critics=2 --init_scale=None --l2_reg=0 --train_every=1 --experience_buffer_length=1000000 --minibatch_size=256 --hidden_layers=[64,32] --gamma=0.995 --cost_gamma=0.995 --layer_norm=True --cost_threshold=5 --beta=0.2 --safe_sac_penalty_max_grad=1000 --clip_gradients=1 --ignore_done_on_timelimit=False --reward_scaling=2 --cost_scaling=2 --record_returns=True
"""

"""
python -m RL.algorithms.safe_sac --env_id=MyPointCircleFinite-v0 --experiment_name=safesac --num_steps_to_run=150000 --normalize_observations=False --alpha=0.2 --actor_learning_rate=0.0001 --learning_rate=0.001 --target_network_update_tau=0.005 --exploit_every=8 --minimum_experience=10000 --logstd_min=-20 --logstd_max=2 --num_critics=2 --init_scale=None --l2_reg=0 --train_every=1 --experience_buffer_length=1000000 --minibatch_size=100 --hidden_layers=[256,256] --gamma=0.99 --cost_gamma=1 --layer_norm=False --cost_threshold=5 --beta=0.2 --safe_sac_penalty_max_grad=1000 --clip_gradients=1 --ignore_done_on_timelimit=False --reward_scaling=2 --cost_scaling=2 --record_returns=False
"""

"""
python -m RL.algorithms.safe_sac --env_id=MyPointCircleFinite-v0 --experiment_name=safesac --num_steps_to_run=150000 --normalize_observations=False --alpha=0.2 --actor_learning_rate=0.0001 --learning_rate=0.001 --target_network_update_tau=0.005 --exploit_every=8 --minimum_experience=10000 --logstd_min=-20 --logstd_max=2 --num_critics=2 --init_scale=None --l2_reg=0 --train_every=1 --experience_buffer_length=1000000 --minibatch_size=100 --hidden_layers=[256,256] --gamma=0.99 --cost_gamma=1 --layer_norm=False --cost_threshold=5 --beta=0.2 --safe_sac_penalty_max_grad=1000 --clip_gradients=1 --ignore_done_on_timelimit=False --reward_scaling=2 --cost_scaling=2 --record_returns=False
"""

"""
python -m RL.algorithms.safe_sac --env_id=Safexp-PointGoal1-v0 --experiment_name=safesac_R20C0.4 --num_steps_to_run=10000000 --normalize_observations=False --alpha=0.2 --actor_learning_rate=0.0001 --learning_rate=0.001 --target_network_update_tau=0.005 --exploit_every=8 --minimum_experience=10000 --logstd_min=-20 --logstd_max=2 --num_critics=2 --init_scale=None --l2_reg=0 --train_every=1 --experience_buffer_length=1000000 --minibatch_size=100 --hidden_layers=[256,256] --gamma=0.99 --cost_gamma=1 --layer_norm=False --cost_threshold=25 --beta=0.2 --safe_sac_penalty_max_grad=1000 --clip_gradients=1 --ignore_done_on_timelimit=False --reward_scaling=20 --cost_scaling=0.4 --record_returns=False --adam_epsilon=0.0001
"""
