import RL
import os
from RL.common.atari_wrappers import wrap_atari
from RL.common.wrappers import MaxEpisodeStepsWrapper
from RL.agents import SeedingAgent, TensorFlowAgent, DQNActAgent, LinearAnnealingAgent, ExperienceBufferAgent, DQNTrainAgent, ParamsCopyAgent, ModelLoaderSaverAgent, EnvRenderingAgent, DQNSensitivityVisualizerAgent, PygletLoopAgent, BasicStatsRecordingAgent, StatsLoggingAgent  # noqa: F401
import gym
from RL.common.utils import need_conv_net


def make(id):
    env = gym.make(id)  # type: gym.Env
    if need_conv_net(env.observation_space):
        env = wrap_atari(env, episode_life=c.atari_episode_life, clip_rewards=c.atari_clip_rewards, framestack_k=c.atari_framestack_k, frameskip_k=c.atari_frameskip_k)
    if 'Lunar' in id:
        env = MaxEpisodeStepsWrapper(env, 600)
    return env


c = RL.Context()
c.set_envs([make(c.env_id) for i in range(1)])
r = RL.Runner(c, "runner", 1e10, num_episodes_to_run=c.n_episodes)

r.register_agent(SeedingAgent(c, "SeedingAgent", seed=c.seed))
r.register_agent(TensorFlowAgent(c, "TensorFlowAgent"))
dqn_act_agent = r.register_agent(DQNActAgent(c, "DQNActAgent"))
r.register_agent(ModelLoaderSaverAgent(c, "LoaderSaverAgent", dqn_act_agent.model.params, os.path.join(RL.logger.get_dir(), 'models'), c.load_model_dir, filename='model'))
if not c.eval_mode:
    r.register_agent(LinearAnnealingAgent(c, "EpsilonAnnealingAgent", 'epsilon', c.set_epsilon, c.minimum_experience, c.epsilon, c.final_epsilon, c.epsilon_anneal_over))
    exp_buff_agent = r.register_agent(ExperienceBufferAgent(c, "ExperienceBufferAgent", nsteps=c.nsteps, buffer_length=c.experience_buffer_length, buffer_size_MB=c.experience_buffer_megabytes))
    dqn_train_agent = r.register_agent(DQNTrainAgent(c, "DQNTrainAgent", dqn_act_agent, exp_buff_agent))
    r.register_agent(ParamsCopyAgent(c, "DQNTargetNetUpdateAgent", dqn_act_agent.model.params, dqn_train_agent.target_model.params, c.target_network_update_every, c.target_network_update_tau))
# if c.sensitivity_visualizer:
#     r.register_agent(DQNSensitivityVisualizerAgent(c, "DQNSensitivityVisualizerAgent", dqn_act_agent.model, 0, auto_dispatch_on_render=False, episode_interval=c.render_interval))
r.register_agent(BasicStatsRecordingAgent(c, "StatsRecordingAgent"))
r.register_agent(StatsLoggingAgent(c, "StatsLoggingAgent", ['Ep Id', 'Ep R', 'Av Ep R (100)', 'Ep L', 'Total Steps', 'Ep Exploit Mode', 'Av Exploit Ep R (100)', 'Total Episodes'], poll_every_episode=1, keys_prefix='Env-0 '))
r.register_agent(StatsLoggingAgent(c, "MiscLoggingAgent", ['epsilon', 'DQNActAgent/main_brain/default/Q_loss'], poll_every_episode=1))
if c.render:
    r.register_agent(EnvRenderingAgent(c, "RenderingAgent", auto_dispatch_on_render=False, episode_interval=c.render_interval))
r.register_agent(PygletLoopAgent(c, "PygletLoopAgent"))

r.run()
