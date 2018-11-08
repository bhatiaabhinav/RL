import gym
import esper


class GymEnv:
    def __init__(self, name):
        self.name = name
        self.env = gym.make(name)
        self.obs = None
        self.done = True
        self.reward = 0
        self.info = {}
        self.action = None


class GymSeedable:
    def __init__(self, seed):
        self.seed = seed
        self.seeding_needed = True


class GymRenderable:
    def __init__(self, mode='human', close=False):
        self.mode = mode


class GymClosable:
    def __init__(self, close=True):
        self.close = close


class BoxActionSpace(gym.spaces.Box):
    pass


class DiscreteActionSpace(gym.spaces.Discrete):
    pass


class BoxObsSpace(gym.spaces.Box):
    pass


class DiscreteObsSpace(gym.spaces.Discrete):
    pass


class SeedingSystem(esper.Processor):
    def __init__(self):
        super().__init__()

    def process(self):
        for ent, (gym_env, gym_seed) in self.world.get_components(GymEnv, GymSeedable):
            if gym_seed.seeding_needed:
                print("seeding env {0} (of entity {1}) with seed {2}".format(gym_env.name, ent, gym_seed.seed))
                gym_env.env.seed(gym_seed.seed)
                gym_seed.seeding_needed = False


class SteppingSystem(esper.Processor):
    def __init__(self):
        super().__init__()

    def process(self):
        for ent, (env) in self.world.get_components(GymEnv):
            if env.obs is None or env.done:
                print("reseting")
                env.obs = env.env.reset()
                env.done = False
            else:
                print("stepping")
                env.obs, env.reward, env.done, env.info = env.env.step(env.action)


class RenderingSystem(esper.Processor):
    def __init__(self):
        super().__init__()

    def process(self):
        for ent, (env, render) in self.world.get_components(GymEnv, GymRenderable):
            print("rendering")
            env.env.render(mode=render.mode)

 
if __name__ == '__main__':
    world = esper.World()
    
    # env entity
    env = world.create_entity(GymEnv("PongNoFrameskip-v4"))
    world.add_component(env, GymSeedable(42))
    world.add_component(env, GymRenderable())

    # processors:
    world.add_processor(SeedingSystem())
    world.add_processor(SteppingSystem())
    world.add_processor(RenderingSystem())
    for i in range(100000):
        world.process()