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

        self.snake_body = [(self.snake_x, self.snake_y)]
        self.snake_length = 1

        self.fruit_x = fruit_x
        self.fruit_y = fruit_y
        self.fruit_width = snake_fruit_width
        self.fruit_height = snake_fruit_height

        self.dx = 20 
        self.dy = 0

        self.current_steps = 0
        self.max_steps = 2000

        self.r_eward = 0

    def snake_and_fruit(self):
        for segment in self.snake_body:
            rec = pygame.Rect(segment[0], segment[1], self.snake_width, self.snake_height)
            pygame.draw.rect(self.screen, GREEN, rec)

        fru1 = pygame.Rect(self.fruit_x, self.fruit_y, self.fruit_width, self.fruit_height)
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

    def _blocked(self, dx, dy):
        nx, ny = self.snake_x + dx, self.snake_y + dy
        if nx < 0 or nx > WIDTH - self.snake_width or ny < 0 or ny > HEIGHT - self.snake_height:
            return 1.0
        if (nx, ny) in self.snake_body[1:]:
            return 1.0
        return 0.0

    def _get_state(self):
        danger = [self._blocked(20, 0), self._blocked(-20, 0),
                self._blocked(0, 20), self._blocked(0, -20)]          # right, left, down, up
        heading = [float(self.dx == 20), float(self.dx == -20),
                float(self.dy == 20), float(self.dy == -20)]
        fruit_dir = [float(self.fruit_x < self.snake_x), float(self.fruit_x > self.snake_x),
                    float(self.fruit_y < self.snake_y), float(self.fruit_y > self.snake_y)]
        return np.array(danger + heading + fruit_dir, dtype=np.float32)

    def self_collide(self):
        # Check if the head's coordinates match any other segment in the body
        if (self.snake_x, self.snake_y) in self.snake_body[1:]:
            return True
        return False
    
    def reset(self):
        self.current_steps = 0
        self.dx, self.dy = 20, 0
        self.snake_x, self.snake_y = SNAKE_X, SNAKE_Y

        self.snake_body = [(self.snake_x, self.snake_y)]
        self.snake_length = 1

        while True:
            self.fruit_x = random.randrange(0, WIDTH - self.fruit_width, 20)
            self.fruit_y = random.randrange(0, HEIGHT - self.fruit_height, 20)
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
            
        if action == 0 and self.dx != -20:
            self.dx = 20
            self.dy = 0
        elif action == 1 and self.dx != 20:
            self.dx = -20
            self.dy = 0
        elif action == 2 and self.dy != -20:
            self.dx = 0
            self.dy = 20
        elif action == 3 and self.dy != 20:
            self.dx = 0
            self.dy = -20

        self.current_steps += 1
                
        # Continually move the snake by adding the current direction values
        self.snake_x += self.dx
        self.snake_y += self.dy

        #add head position to the body list
        self.snake_body.insert(0, (self.snake_x, self.snake_y))

        step_reward = self.fruit_collid()

        # If fruit was eaten, increase the target length
        if step_reward > 0:
            self.snake_length += 1
            
        # Remove the tail if the body is longer than the target length
        if len(self.snake_body) > self.snake_length:
            self.snake_body.pop()

        terminate = bool(self.wall_collid() or self.self_collide())

        observation = self._get_state()

        prev_dist = abs(self.fruit_x - self.snake_x) + abs(self.fruit_y - self.snake_y)
        new_dist = abs(self.fruit_x - self.snake_x) + abs(self.fruit_y - self.snake_y)

        if terminate:
            reward = -1.0
        elif step_reward > 0:
            reward = step_reward
        else:
            reward = 0.05 if new_dist < prev_dist else -0.05

        truncated =  bool(self.current_steps > self.max_steps)

        self.clock.tick(1000)
            
        return observation, reward, terminate, truncated


