import RL
import RL.envs  # noqa: F401
import gym
import sys


env_id = sys.argv[1]
e = gym.make(env_id)  # type: gym.Env

print("Observation space:", e.observation_space, e.observation_space.low, e.observation_space.high)
print("action_space:", e.action_space, e.action_space.low, e.action_space.high)

r = 0
d = True
info = {}
render = True

for i in range(10000):
    if d:
        obs = e.reset()
        if render:
            e.render()
    obs, r, d, info = e.step(e.action_space.sample())
