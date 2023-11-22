# Reinforcement Learning with REINFORCE Algorithm

Welcome to the repository for the implementation of the REINFORCE algorithm from scratch using Python and Numpy, with a focus on solving the OpenAI Gym CartPole environment. This project consists of three main files:

1. **DNNScratch.py**: This file contains the implementation of a simple Neural Network from scratch using the powerful Numpy library. The neural network serves as the policy network in the REINFORCE algorithm.

2. **ReinforceLearning02.py**: This file is the core implementation of the REINFORCE algorithm from scratch. It utilizes the neural network implemented in `DNNScratch.py` to learn a policy for the CartPole environment.

3. **TensorFlowTestings.py**: This file provides an alternative implementation of the REINFORCE algorithm using TensorFlow. The purpose of this file is to compare the performance and accuracy of the handcrafted implementation with the widely-used TensorFlow library.

## REINFORCE Algorithm Overview

The REINFORCE algorithm is a policy gradient method used in reinforcement learning. It is designed to learn a policy for an agent to maximize the expected cumulative reward. The key idea is to update the policy in the direction that increases the probability of actions that lead to higher rewards.

### Components of REINFORCE Algorithm

- **Policy Network**: The neural network implemented in `DNNScratch.py` serves as the policy network. It takes the state of the environment as input and outputs the probability distribution over possible actions.

- **Policy Gradient**: REINFORCE employs the policy gradient method, which involves taking the gradient of the expected reward with respect to the policy parameters. This gradient is then used to update the policy in the direction that maximizes expected cumulative reward.

- **Monte Carlo Sampling**: The algorithm estimates the expected gradient using Monte Carlo sampling. Trajectories are sampled, and the expected reward is computed to update the policy.

## Files Description

### 1. DNNScratch.py

This file contains the implementation of a simple neural network using Numpy. The neural network architecture is designed for the policy network in the REINFORCE algorithm.

### 2. ReinforceLearning02.py

The core implementation of the REINFORCE algorithm. It uses the neural network from `DNNScratch.py` to learn a policy for the OpenAI Gym CartPole environment.

### 3. TensorFlowTestings.py

An alternative implementation of the REINFORCE algorithm using TensorFlow. This file is included for comparison purposes to evaluate the performance of the handcrafted implementation against a widely-used deep learning library.

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/your-username/REINFORCE-CartPole.git
cd REINFORCE-CartPole
```

2. Run the `ReinforceLearning02.py` file to train the REINFORCE algorithm:

```bash
python ReinforceLearning02.py
```

3. Optionally, run the `TensorFlowTestings.py` file to compare the performance with the TensorFlow implementation:

```bash
python TensorFlowTestings.py
```

## Dependencies

- Numpy
- OpenAI Gym
- TensorFlow (only required for `TensorFlowTestings.py`)

## Acknowledgments

- OpenAI for the Gym toolkit and CartPole environment.
- The community and contributors of Numpy and TensorFlow.

Feel free to explore, experiment, and extend the code. If you have any questions or suggestions, please open an issue. Happy coding!
