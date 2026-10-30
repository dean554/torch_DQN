import pygame
import random
import numpy as np

import os
#os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()

WIDTH = 500
HEIGHT = 500

#colors
BLACK = 0
GREEN = (0, 200, 0)
RED = (200, 0, 0)

#snake and fruit start
SNAKE_X , SNAKE_Y = WIDTH//2, HEIGHT//2
FRUIT_X , FRUIT_Y = (WIDTH//2) + 100, (HEIGHT//2) + 50


#snake and fruit size
SNAKE_FRUIT_WIDTH, SNAKE_FRUIT_HEIGHT = 20, 20

class s_env:
    def __init__(self, snake_x, snake_y, fruit_x, fruit_y, snake_fruit_width, snake_fruit_height):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("game environment")
        self.clock = pygame.time.Clock()

        self.snake_x = snake_x
        self.snake_y = snake_y
        self.snake_width = snake_fruit_width
        self.snake_height = snake_fruit_height

        self.state1 = self.snake_x
        self.state2 = self.snake_y

        self.fruit_x = fruit_x
        self.fruit_y = fruit_y
        self.fruit_width = snake_fruit_width
        self.fruit_height = snake_fruit_height

        self.dx = 10 
        self.dy = 0

        self.current_steps = 0
        self.max_steps = 200

        self.r_eward = 0

    def snake_and_fruit(self):
        rec1 = pygame.Rect(self.snake_x, self.snake_y, self.snake_width, self.snake_height)
        fru1 = pygame.Rect(self.fruit_x, self.fruit_y, self.fruit_width, self.fruit_height)
        pygame.draw.rect(self.screen, GREEN, rec1)
        pygame.draw.rect(self.screen, RED, fru1)


    def wall_collid(self):
        if self.snake_x > WIDTH - self.snake_width or self.snake_x < 0:
            return True
        if self.snake_y > HEIGHT - self.snake_height or self.snake_y < 0:
            return True

        return False

    def fruit_collid(self):
        snake_rect = pygame.Rect(self.snake_x, self.snake_y, self.snake_width, self.snake_height)
        fruit_rect = pygame.Rect(self.fruit_x, self.fruit_y, self.fruit_width, self.fruit_height)

        if snake_rect.colliderect(fruit_rect):
            self.fruit_x = random.randrange(0, WIDTH - self.fruit_width, 20)
            self.fruit_y = random.randrange(0, HEIGHT - self.fruit_height, 20)

            return 1.0 # Return reward only when eaten
        return 0.0

    def _get_state(self):
        dx = (self.fruit_x - self.snake_x) / WIDTH
        dy = (self.fruit_y - self.snake_y) / HEIGHT
        sx = self.snake_x / WIDTH
        sy = self.snake_y / HEIGHT

        wall_left   = self.snake_x / WIDTH
        wall_right  = (WIDTH - self.snake_x - self.snake_width) / WIDTH
        wall_top    = self.snake_y / HEIGHT
        wall_bottom = (HEIGHT - self.snake_y - self.snake_height) / HEIGHT
        
        return np.array([dx, dy, sx, sy, wall_left, wall_right, wall_top, wall_bottom], dtype=np.float32)
        
    def reset(self):
        self.current_steps = 0
        self.snake_x, self.snake_y = SNAKE_X, SNAKE_Y

        while True:
            self.fruit_x = random.randrange(0, WIDTH - self.fruit_width, 10)
            self.fruit_y = random.randrange(0, HEIGHT - self.fruit_height, 10)
            if self.fruit_x != self.snake_x or self.fruit_y != self.snake_y:
                break

        self.r_eward = 0
        return self._get_state()
        
    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        self.screen.fill((BLACK, BLACK, BLACK))

        #draw snake rect
        self.snake_and_fruit()

        pygame.display.flip()

        self.fruit_collid()
            
        if action == 0:
            self.dx = 10
            self.dy = 0
        elif action == 1:
            self.dx = -10
            self.dy = 0
        elif action == 2:
            self.dx = 0
            self.dy = 10
        elif action == 3:
            self.dx = 0
            self.dy = -10

        self.current_steps += 1
                
        # Continually move the snake by adding the current direction values
        self.snake_x += self.dx
        self.snake_y += self.dy
        self.state1 = self.snake_x
        self.state2 = self.snake_y

        step_reward = self.fruit_collid()
        terminate = bool(self.wall_collid())

        observation = self._get_state()

        if terminate:
            reward = -1.0
        else:
            reward = step_reward
        truncated =  bool(self.current_steps > self.max_steps)
        
        self.clock.tick(600)
            
        return observation, reward, terminate, truncated


