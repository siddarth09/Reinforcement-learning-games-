

# 🚕 Taxi Driver Reinforcement Learning (DQN)

This project implements a **Deep Q-Network (DQN)** agent to solve the classic **Taxi-v3** environment from Gymnasium. The goal is to learn an optimal policy for picking up and dropping off passengers in a grid-based city while minimizing penalties and travel cost.

Unlike policy-gradient methods (PPO), this problem is solved using **value-based reinforcement learning** with discrete actions.

<video width="640" height="360" controls autoplay loop muted>
  <source src="../assets/taxi-driver.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

##  Problem Overview

The Taxi-v3 environment is a **finite Markov Decision Process** with:

* A **discrete state space** of size 500
* A **discrete action space** of size 6

The agent must:

1. Navigate the taxi to the passenger
2. Pick up the passenger
3. Navigate to the destination
4. Drop off the passenger correctly

Incorrect actions (illegal pickup/drop-off) incur heavy penalties.

---

##  Observation Space (State Representation)

The environment internally encodes the state as a single integer:

```
state ∈ {0, 1, ..., 499}
```

This integer encodes:

* Taxi row
* Taxi column
* Passenger location
* Destination

### One-Hot Encoding

To make this compatible with a neural network, the state is converted into a **one-hot vector**:

```
state_dim = 500
```

Example:

```
state = 123
→ one_hot(state) ∈ R^500
```

This allows the Q-network to approximate the action-value function over all possible states.

---

##  Action Space

The action space is **discrete**:

| Action | Meaning           |
| -----: | ----------------- |
|      0 | Move South        |
|      1 | Move North        |
|      2 | Move East         |
|      3 | Move West         |
|      4 | Pickup Passenger  |
|      5 | Dropoff Passenger |

The policy selects the action with the **highest predicted Q-value**.

---

##  Q-Network Architecture

The Q-function is approximated using a **Multi-Layer Perceptron (MLP)**:

```text
Input:  500-dim one-hot state
Hidden: 128 → ReLU
Hidden: 128 → ReLU
Output: Q-values for 6 actions
```

### PyTorch Model

```python
Q(s) = QNetwork(state)
action = argmax_a Q(s, a)
```

This network learns to estimate the expected discounted return of each action from a given state.

---

## 🗂 Experience Replay Buffer

To stabilize training, experiences are stored in a **Replay Buffer**:

Each transition stored as:

```
(s, a, r, s', done)
```

During training:

* Random mini-batches are sampled
* Temporal correlations between consecutive transitions are broken
* Training becomes more stable and data-efficient

---

##  Learning Algorithm: Deep Q-Network (DQN)

The learning update follows the Bellman equation:

```
Q(s, a) ← r + γ max_a' Q_target(s', a')
```

### Training Details

* Loss Function: Mean Squared Error (MSE)
* Optimizer: Adam
* Discount Factor (γ): 0.99
* Target Network: Updated periodically for stability

### Training Step Summary

1. Sample a batch from replay buffer
2. Compute current Q(s, a)
3. Compute target Q-value using target network
4. Minimize MSE loss
5. Backpropagate and update Q-network

---

##  Exploration Strategy (Epsilon-Greedy)

Exploration is handled using **epsilon-greedy action selection**:

* With probability ε: choose a random action
* With probability (1 − ε): choose the greedy action

Epsilon decays over time:

```text
ε_start = 1.0
ε_min   = 0.05
ε_decay = 0.999
```

This ensures:

* Early exploration
* Gradual exploitation of learned policy

---

##  Training Loop Summary

* Environment: Taxi-v3
* Episodes: ~5000
* Target network update: every 50 episodes
* Replay buffer size: 100,000 transitions

At the end of training, the learned policy is saved as:

```text
taxi_dqn.pth
```

---

## ▶️ Policy Evaluation

During testing:

* The trained Q-network is loaded
* The agent acts **deterministically**
* The environment is rendered using Gym’s human mode

The agent reliably:

* Navigates to the passenger
* Picks up correctly
* Delivers to the destination
* Avoids illegal actions




