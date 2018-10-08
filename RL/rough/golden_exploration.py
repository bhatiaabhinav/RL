import numpy as np
import pygame
import sobol_seq
import sys


def new_random_point(random: np.random.RandomState):
    x = random.rand()
    y = random.rand()
    return x, y


def new_golden_random_point(last_x, last_y):
    addition_x = (np.sqrt(3) - 1)
    addition_y = np.sqrt(2) - 1
    x, y = last_x + addition_x, last_y + addition_y
    while x > 1:
        x = x - 1
    while y > 1:
        y = y - 1
    return x, y


def new_sobol_random_point(sobol_seed):
    vec, sobol_seed = sobol_seq.i4_sobol(2, sobol_seed)
    return vec[0], vec[1], sobol_seed


def draw_new_point(x, y, side, screen):
    pygame.draw.rect(screen, (0, 128, 255), pygame.Rect(
        x - side / 2, y - side / 2, side, side))


def main():
    pygame.init()
    seed = 0
    sobol_seed = seed
    random = np.random.RandomState(seed)
    if len(sys.argv) > 1 and sys.argv[1] in ['random', 'sobol', 'golden']:
        mode = sys.argv[1]
    else:
        print('No or unrecognirzed arguments were given. Falling back to random mode')
        mode = 'random'
    if mode == 'sobol':
        x, y, sobol_seed = new_sobol_random_point(sobol_seed)
    else:
        x, y = new_random_point(random)

    screen = pygame.display.set_mode((800, 400))
    FPS = 30
    clock = pygame.time.Clock()  # Create a clock object
    should_quit = False

    count = 0
    while not should_quit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_quit = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                draw_new_point(x * 800, y * 400, 10, screen)
                count += 1
                print(count)
                if mode == 'golden':
                    x, y = new_golden_random_point(x, y)
                elif mode == 'sobol':
                    x, y, sobol_seed = new_sobol_random_point(sobol_seed)
                else:
                    x, y = new_random_point(random)

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == '__main__':
    main()
