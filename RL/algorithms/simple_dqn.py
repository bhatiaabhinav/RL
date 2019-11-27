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
from RL.common.wrappers import wrap_standard

from RL.contexts import DQNContext

c = DQNContext()
c.set_env(wrap_standard(gym.make(c.env_id), c))

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


"""
To run this algorithm in bash: do:

python -m RL.algorithms.simple_dqn --env_id=BreakoutNoFrameskip-v4 --experiment_name=SimpleDQN --double_dqn=False --dueling_dqn=False --experience_buffer_length=100000 --atari_clip_rewards=False --atari_episode_life=True --learning_rate=1e-4 --convs="[(16,8,4),(32,4,2),(32,3,1)]" --hidden_layers="[256]" --normalize_observations=False --minimum_experience=10000 --target_network_update_every=2000
"""
