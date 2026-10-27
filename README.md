# Torch DQN: Custom 1D Environment

This repository contains a lightweight Deep Q-Network (DQN) implementation adapted directly from the [Official PyTorch DQN Tutorial]
(https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html). 

It modifies the official boilerplate to train an agent on a custom 1D continuous-state environment ( 'envi.py ') designed as a simplified alternative to standard benchmarks like CartPole.

## Environment Dynamics ( 'envi.py ')
The custom environment simulates a 1D state variable with the following mechanics:
* **Initial State:** The agent always starts at position  '10.0 '.
* **Boundaries:** The episode terminates if the state drops below  '5.0 ' or exceeds  '15.0 '.
* **Action Space:** Discrete binary actions where  '0 ' decreases the state by  '0.5 ' and  '1 ' increases it by  '0.5 '.
* **Noise:** Uniform random noise between  '-0.8 ' and  '0.8 ' is applied to the state after every action .
* **Reward Structure:** The agent receives a reward of  '1.0 ' for every step survived, dropping to  '0.0 ' upon out-of-bounds termination.
* **Truncation:** Episodes are forcibly truncated after reaching  '200 ' steps .

## DQN Architecture & Tutorial Modifications ( 'dqn.py ')
The neural network relies on a feedforward architecture ( 'input -> 128 -> 128 -> output ') optimized via Huber loss, an epsilon-greedy exploration strategy, and Polyak averaging for target network updates ( 'TAU = 0.001 '). 

To support the custom  'envi.py ' logic instead of standard  'gymnasium ' environments, the following specific modifications were implemented to the tutorial's baseline code:
* **Environment Instantiation:** Replaced the standard  'gym.make() ' call with the local  'envirn() ' class initialization .
* **Observation Space Sizing:** Replaced gym observation space queries with a direct array length calculation ( 'observation = len(state) ') to define the neural network's input dimension .
* **Tensor Formatting:** Adjusted the state transition loop to wrap the raw  'envi.py ' NumPy arrays directly into 2D PyTorch tensors ( 'torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) ') to satisfy the batch dimension requirements of the  'policy_net ' .

The script automatically selects hardware acceleration (CUDA/MPS) if available and generates live Matplotlib training duration plots matching the official tutorial's visualization style