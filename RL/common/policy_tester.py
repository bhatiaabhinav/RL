import gym
import numpy as np


def test(env: gym.Env, actor, seed, episodes, render):
    print('Evaluating {0} episodes. Seed={1}'.format(episodes, seed))
    env.seed(seed)
    Rs = []
    for ep in range(episodes):
        d, R = False, 0
        obs = env.reset()
        if render:
            env.render()
        while not d:
            action = actor.act([obs])[0]
            # print(action)
            obs, r, d, _ = env.step(action)
            if render:
                env.render()
            # time.sleep(1/60)
            R += r
        Rs.append(R)
        print('Episode {0}\tReward: {1}\tAverage_Reward: {2}'.format(
            ep, R, np.average(Rs)))
    print('---------------------------------------------------')
    av = np.average(Rs) if episodes > 0 else 0
    print('Testing Finished')
    print('Average reward per episode: {0}'.format(av))
    return av
