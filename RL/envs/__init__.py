import gym

gym.register('AntRLLab-v0', entry_point='RL.envs.mujoco.ant_env:AntEnv', max_episode_steps=1000)
gym.register('AntGatherRLLab-v0', entry_point='RL.envs.mujoco.gather.ant_gather_env:AntGatherEnv', max_episode_steps=1000)
gym.register('PointGatherRLLab-v0', entry_point='RL.envs.mujoco.gather.point_gather_env:PointGatherEnv', max_episode_steps=1000)
gym.register('PointRLLab-v0', entry_point='RL.envs.mujoco.point_env:PointEnv', max_episode_steps=50)
gym.register('SafeAntRLLab-v0', entry_point='RL.envs.mujoco_safe.ant_env_safe:SafeAntEnv', max_episode_steps=100)
gym.register('SafePointRLLab-v0', entry_point='RL.envs.mujoco_safe.point_env_safe:SafePointEnv', max_episode_steps=50)