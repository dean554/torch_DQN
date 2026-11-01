from game import s_env
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import random
from collections import namedtuple, deque
from itertools import count

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'cpu')

Trabsitions = namedtuple('Transitions', (
        'state', 'action', 'next_state', 'reward')
    )

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        return self.memory.append(Trabsitions(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.l1  = nn.Linear(n_observations, 256)
        self.l2  = nn.Linear(256, 256)
        self.l3  = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.001
LR = 1e-3

#snake and fruit start
WIDTH = 500
HEIGHT = 500
SNAKE_X , SNAKE_Y = WIDTH//2, HEIGHT//2
FRUIT_X , FRUIT_Y = (WIDTH//2) + 100, (HEIGHT//2) + 50


#snake and fruit size
SNAKE_FRUIT_WIDTH, SNAKE_FRUIT_HEIGHT = 20, 20

env = s_env(SNAKE_X, SNAKE_Y, FRUIT_X , FRUIT_Y, SNAKE_FRUIT_WIDTH, SNAKE_FRUIT_HEIGHT)

state = env.reset()
n_state = len(state)
n_actions = 4

memory = ReplayMemory(10000)

policy_net = DQN(n_state, n_actions).to(device)
target_net = DQN(n_state, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

steps_done = 0

def select_action(state):
    global steps_done
    steps_done += 1
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    math.exp(-1. * steps_done / EPS_DECAY)

    if sample > eps_threshold:
        return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randint(0, 3)]], device=device, dtype=torch.long)




def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    batch = Trabsitions(*zip(*transitions))


    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_state).max(1).values

    expected_next_state_values = (GAMMA * next_state_values) + reward_batch

    #loss
    criterian = nn.SmoothL1Loss()
    loss = criterian(state_action_values, expected_next_state_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 60000
else:
    num_episodes = 2000

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    episode_reward = 0.0
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated = env.step(action.item())
        episode_reward += reward
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            torch.save(policy_net.state_dict(), "snake_dqn.pth")
            print(f"Model saved to snake_dqn.pth. {reward}")
            break

print('Complete')
