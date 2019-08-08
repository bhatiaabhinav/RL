import gym

gym.register('AntRLLab-v0', entry_point='RL.envs.mujoco.ant_env:AntEnv', max_episode_steps=1000)
gym.register('AntGatherRLLab-v0', entry_point='RL.envs.mujoco.gather.ant_gather_env:AntGatherEnv', max_episode_steps=1000)
gym.register('SafePointGatherRLLab-v0', entry_point='RL.envs.mujoco.gather.point_gather_env:PointGatherEnv', max_episode_steps=15)
gym.register('PointRLLab-v0', entry_point='RL.envs.mujoco.point_env:PointEnv', max_episode_steps=50)
gym.register('SafeAntRLLab-v0', entry_point='RL.envs.mujoco_safe.ant_env_safe:SafeAntEnv', max_episode_steps=100)
gym.register('SafePointRLLab-v0', entry_point='RL.envs.mujoco_safe.point_env_safe:SafePointEnv', max_episode_steps=50)

gym.register(
    id='MyPointCircle-v0',
    entry_point='RL.envs.my_point_circle:SimplePointEnv',
    max_episode_steps=50
)

gym.register(
    id='MyPointCircleFinite-v0',
    entry_point='RL.envs.my_point_circle:SimplePointEnv',
    kwargs={"horizon": 50}
)

gym.register(
    id='HalfCheetahSafe-v2',
    entry_point='RL.envs.half_cheetah_safe:HalfCheetahSafeEnv',
    kwargs={'max_speed': 1.0},
    max_episode_steps=200
)

gym.register(
    id='HalfCheetahSafeFinite-v2',
    entry_point='RL.envs.half_cheetah_safe:HalfCheetahSafeEnv',
    kwargs={'max_speed': 1.0, 'horizon': 200}
)
