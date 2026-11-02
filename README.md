# Torch DQN: Custom Environments

This repository contains lightweight Deep Q-Network (DQN) implementations adapted from the [Official PyTorch DQN Tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

The tutorial's boilerplate is modified to train agents on custom environments instead of standard `gymnasium` benchmarks like CartPole. There are two projects:

| Folder | Environment | Description |
|--------|-------------|-------------|
| [`v1/`](./v1) | Custom 1D continuous-state environment | A simplified alternative to CartPole |
| [`snake/`](./snake) | Pygame Snake game | A DQN agent that learns to play Snake |

## Repository Structure

```
torch_DQN/
├── v1/
│   ├── envi.py          # 1D environment
│   └── dqn.py           # DQN training script
├── snake/
│   ├── game.py          # Pygame Snake environment (s_env)
│   ├── dqn.py           # Training script (saves snake_dqn.pth)
│   ├── dqn_plot.py      # Training script with live Matplotlib reward plot
│   └── test.py          # Watch a trained agent play
├── snake_dqn.pth        # Saved weights of the trained snake agent
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
git clone https://github.com/dean554/torch_DQN.git
cd torch_DQN
pip install -r requirements.txt
```

Make sure `requirements.txt` includes `torch`, `numpy`, `matplotlib` and `pygame` (the Snake project needs `pygame`).

---

## Project 1: Custom 1D Environment (`v1/`)

### Environment Dynamics (`envi.py`)

The custom environment simulates a 1D state variable with the following mechanics:

- **Initial State:** The agent always starts at position `10.0`.
- **Boundaries:** The episode terminates if the state drops below `5.0` or exceeds `15.0`.
- **Action Space:** Discrete binary actions where `0` decreases the state by `0.5` and `1` increases it by `0.5`.
- **Noise:** Uniform random noise between `-0.8` and `0.8` is applied to the state after every action.
- **Reward Structure:** The agent receives a reward of `1.0` for every step survived, dropping to `0.0` upon out-of-bounds termination.
- **Truncation:** Episodes are forcibly truncated after reaching `200` steps.

### DQN Architecture & Tutorial Modifications (`dqn.py`)

The neural network relies on a feedforward architecture (`input -> 128 -> 128 -> output`) optimized via Huber loss, an epsilon-greedy exploration strategy, and Polyak averaging for target network updates (`TAU = 0.001`).

To support the custom `envi.py` logic instead of standard `gymnasium` environments, the following modifications were made to the tutorial's baseline code:

- **Environment Instantiation:** Replaced the standard `gym.make()` call with the local `envirn()` class initialization.
- **Observation Space Sizing:** Replaced gym observation space queries with a direct array length calculation (`observation = len(state)`) to define the neural network's input dimension.
- **Tensor Formatting:** Adjusted the state transition loop to wrap the raw `envi.py` NumPy arrays directly into 2D PyTorch tensors (`torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)`) to satisfy the batch dimension requirements of the `policy_net`.

The script automatically selects hardware acceleration (CUDA/MPS) if available and generates live Matplotlib training duration plots matching the official tutorial's visualization style.

### Usage

```bash
cd v1
python dqn.py
```

---

## Project 2: Snake DQN (`snake/`)

A DQN agent that learns to play the classic Snake game, built with Pygame and trained with the same tutorial-based DQN loop as `v1`.

### Environment Dynamics (`game.py`)

The `s_env` class implements a Pygame Snake game with the following mechanics:

- **Board:** A `500 x 500` window. The snake and fruit are `20 x 20` blocks, and the snake moves `10` pixels per step.
- **Initial State:** The snake starts in the center of the board heading right. The fruit spawns at a random position.
- **Action Space:** Four discrete actions: `0` = right, `1` = left, `2` = down, `3` = up. Actions that would reverse the snake directly into itself are ignored.
- **Observation (12 binary features):**
  - *Danger* (4): whether moving right / left / down / up would hit a wall or the snake's body
  - *Heading* (4): the snake's current direction of travel
  - *Fruit direction* (4): whether the fruit is to the left / right / above / below the snake's head
- **Growth:** Eating the fruit grows the snake by one segment and respawns the fruit.
- **Termination:** The episode ends when the snake hits a wall or its own body.
- **Truncation:** Episodes are truncated after `2000` steps.
- **Reward Structure:**

| Event | Reward |
|-------|--------|
| Eat fruit | `+1.0` |
| Hit wall or self | `-1.0` |
| Any other step | small shaping reward of `±0.05` |

### DQN Architecture & Hyperparameters

| Setting | Value |
|---------|-------|
| Network (`dqn.py`, `dqn_plot.py`) | `12 -> 256 -> 256 -> 4` (ReLU) |
| Loss / optimizer | Huber (SmoothL1) loss, AdamW (`LR = 1e-3`) |
| Discount (`GAMMA`) | `0.99` |
| Batch size | `128` |
| Replay memory | `10,000` transitions |
| Exploration | Epsilon-greedy, `0.9 -> 0.01`, decay `2500` steps |
| Target update | Polyak averaging, `TAU = 0.001` |
| Episodes | `60,000` with CUDA/MPS, otherwise `2,000` |

As in `v1`, the input size is taken directly from the environment (`n_state = len(state)`) and states are wrapped into 2D tensors with `.unsqueeze(0)`.

### Scripts

- **`dqn.py`**: Trains the agent. The policy network weights are saved to `snake_dqn.pth` at the end of every episode.
- **`dqn_plot.py`**: Same training loop, but plots the reward of each episode live with Matplotlib, along with a 100-episode moving average. It does not save the model.
- **`test.py`**: Loads a trained model and plays up to `2000` games with a greedy policy (no exploration), printing the fruits eaten per episode and the average at the end.

### Usage

Train the agent:

```bash
cd snake
python dqn.py
```

Train with a live reward plot:

```bash
cd snake
python dqn_plot.py
```

Watch a trained agent play (pass the model path, or it defaults to `snake/snake_dqn.pth`):

```bash
python snake/test.py snake_dqn.pth
```

Close the Pygame window to stop the test run.

---

## Notes

- Both projects automatically select hardware acceleration when available.
- Training progress is plotted live with Matplotlib (`v1/dqn.py` and `snake/dqn_plot.py`).