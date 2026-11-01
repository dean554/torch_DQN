import sys
import time

import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

from game import s_env

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "snake_dqn.pth"
N_EPISODES = 2000 # how many games to play
STEP_DELAY = 0.04 # seconds between moves

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(n_observations, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


# Same start positions and sizes as in training
WIDTH, HEIGHT = 500, 500
SNAKE_X, SNAKE_Y = WIDTH // 2, HEIGHT // 2
FRUIT_X, FRUIT_Y = (WIDTH // 2) + 100, (HEIGHT // 2) + 50
SIZE = 20

env = s_env(SNAKE_X, SNAKE_Y, FRUIT_X, FRUIT_Y, SIZE, SIZE)

state = env.reset()
n_state = len(state)     # input size comes from the environment, as in training
n_actions = 4

# Build the network, load the saved weights, switch to evaluation mode
policy_net = DQN(n_state, n_actions).to(device)
policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
policy_net.eval()
print(f"Loaded {MODEL_PATH} (state size {n_state}, device {device})")

scores = []
quit_requested = False

for episode in range(N_EPISODES):
    state = env.reset()
    fruits_eaten = 0
    done = False

    while not done and not quit_requested:
        # Let the user close the window (peek does not remove the event)
        if pygame.event.peek(pygame.QUIT):
            quit_requested = True
            break

        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # Greedy action: always pick the best Q-value (no random exploration)
        with torch.no_grad():
            action = policy_net(state_t).argmax(dim=1).item()

        state, reward, terminated, truncated = env.step(action)
        if reward > 0:
            fruits_eaten += 1
        done = terminated or truncated

        if STEP_DELAY > 0:
            time.sleep(STEP_DELAY)

    if quit_requested:
        break

    scores.append(fruits_eaten)
    print(f"Episode {episode + 1}: fruits eaten = {fruits_eaten}")

if scores:
    print(f"\nAverage fruits per episode over {len(scores)} games: {sum(scores) / len(scores):.2f}")

pygame.quit()