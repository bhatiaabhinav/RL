import gym
import numpy as np
import pygame


class SimplePointEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, mass=1, size=40, target_dist=15, xlim=2.5, dt=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size  # size of viewport will be 2size x 2size
        self.target_dist = target_dist
        self.xlim = xlim
        self.dt = dt
        self.positon = np.zeros(2)  # x, y
        self.orientation = 0  # angle with x axis
        self.velocity = np.zeros(2)  # vx, vy
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.get_observation().shape, dtype=np.float32)
        self.screen_width = 800
        self.screen_height = 800
        self.viewer = None
        self.close_viewer = False
        self.random = np.random.RandomState()

    def stimulate(self, steps, dt):
        for step_id in range(steps):
            self.positon += self.velocity * dt
            self.positon = np.clip(self.positon, -self.size, self.size)

    def get_observation(self):
        return np.concatenate((self.positon, self.velocity, [np.cos(self.orientation), np.sin(self.orientation)]))

    def get_reward(self):
        x, y = self.positon
        vx, vy = self.velocity
        reward = -y * vx + x * vy
        reward /= (1 + np.abs(np.sqrt(x**2 + y**2) - self.target_dist))
        return reward

    def get_done(self):
        return False

    def get_inf0(self):
        return {}

    def seed(self, seed):
        self.random.seed(seed)

    def reset(self):
        self.positon = self.random.standard_normal(size=[2]) * 0.01
        self.velocity = self.random.standard_normal(size=[2]) * 0.1
        self.orientation = np.arctan(self.velocity[1] / self.velocity[0])
        return self.get_observation()

    def step(self, action):
        action = np.clip(action, -1, 1)
        self.orientation += action[1]
        # compute increment in each direction
        vx = np.cos(self.orientation) * action[0]
        vy = np.sin(self.orientation) * action[0]
        self.velocity = np.array([vx, vy])
        self.positon += self.velocity * self.dt
        self.positon = np.clip(self.positon, -self.size, self.size)
        obs = self.get_observation()
        r = self.get_reward()
        d = self.get_done()
        info = self.get_inf0()
        info['Safety_reward'] = -float(np.abs(self.positon[0]) >= self.xlim)
        return obs, r, d, info

    def create_window(self):
        pygame.init()
        self.viewer = pygame.display.set_mode(size=(self.screen_width, self.screen_height))
        pygame.display.set_caption(self.spec.id)
        self.close_viewer = False

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close_viewer = True

    def to_screen_coords(self, game_coords):
        # game coords are cartesian. have origin at center
        # small screen coords are according to 2size x 2size screen
        screen_x_small = self.size + game_coords[0]
        screen_y_small = self.size - game_coords[1]
        # final screen coords according to width and height
        screen_x = int(screen_x_small * self.screen_width / (2 * self.size))
        screen_y = int(screen_y_small * self.screen_height / (2 * self.size))
        return (screen_x, screen_y)

    def scale_size(self, size):
        w = int(size[0] * self.screen_width / (2 * self.size))
        h = int(size[1] * self.screen_height / (2 * self.size))
        return (w, h)

    def to_screen_rect(self, rect):
        return [*self.to_screen_coords(rect[0:2]), *self.scale_size(rect[2:4])]

    def draw(self):
        # clear background
        WHITE = (0xFF, 0xFF, 0xFF)
        GREEEN = (0x00, 0xFF, 0x00)
        BLUE = (0x00, 0x00, 0xFF)
        RED = (0xFF, 0x00, 0x00)
        self.viewer.fill(WHITE)
        # draw circle
        pygame.draw.ellipse(self.viewer, GREEEN, [*self.to_screen_coords([-self.target_dist, self.target_dist]), *self.scale_size([2 * self.target_dist, 2 * self.target_dist])], int(1 * self.screen_width / (2 * self.size)))
        # draw restrictions:
        pygame.draw.rect(self.viewer, BLUE, self.to_screen_rect([-self.xlim - 1, self.size, 2, 2 * self.size]))
        pygame.draw.rect(self.viewer, BLUE, self.to_screen_rect([self.xlim - 1, self.size, 2, 2 * self.size]))
        # draw point
        pygame.draw.ellipse(self.viewer, RED, [*self.to_screen_coords(self.positon - 0.5), *self.scale_size([1, 1])])
        # flip
        pygame.display.flip()

    def render(self, mode='human'):
        assert mode in self.metadata['render.modes']
        # create window:
        if self.viewer is None:
            self.create_window()
        # events
        self.process_events()
        # draw:
        self.draw()
        # respect the mode:
        if mode == 'human':
            pass
        elif mode == 'rgb_array':
            raise NotImplementedError()
        # close if pressed close button:
        if self.close_viewer:
            self.close()

    def close(self):
        pygame.display.quit()
        pygame.quit()
        self.viewer = None
        self.close_viewer = False
