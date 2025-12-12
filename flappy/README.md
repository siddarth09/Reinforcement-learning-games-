# 🐦 Flappy Bird Reinforcement Learning (Tabular Q-Learning)

This project implements a **tabular Q-learning agent** to solve a custom **Flappy Bird environment** built using **Pygame**.

Unlike deep reinforcement learning approaches (DQN, PPO), this project demonstrates how **classic Q-learning** can successfully solve a continuous, physics-based game by **discretizing the state space**.

<video width="640" height="360" controls autoplay loop muted>
  <source src="../assets/flappy.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

##  Problem Overview

The goal is to train an agent to control a bird that must fly through gaps between moving pipes without colliding with them or the screen boundaries.

At every timestep, the agent must decide whether to:

* **Flap** (apply upward velocity), or
* **Do nothing** (let gravity act)

The challenge lies in:

* Continuous dynamics (gravity, velocity)
* Sparse and delayed rewards
* Precise timing required to pass through pipe gaps

---

##  Environment Design

The environment simulates:

* Gravity-based vertical motion
* Horizontal moving pipes
* Collision detection
* Scoring for successful pipe passes

Each episode ends when:

* The bird hits a pipe, or
* The bird hits the ground or ceiling

---

##  Observation Space (State Representation)

The environment state is continuous, but **Q-learning requires a finite state space**.

To bridge this gap, the state is **discretized** into three components:

### 1. Vertical distance to pipe gap center (dy)

```
dy = bird_y − pipe_gap_center
```

### 2. Horizontal distance to next pipe (dx)

```
dx = pipe_x − bird_x
```

### 3. Bird vertical velocity (v)

Each component is discretized into bins:

| Component | Bins |
| --------- | ---- |
| dy        | 20   |
| dx        | 20   |
| velocity  | 10   |

### Final State Tuple

```
state = (dy_bin, dx_bin, velocity_bin)
```

This results in a **finite state space** suitable for tabular Q-learning.

---

##  Action Space

The action space is **discrete**:

| Action | Meaning     |
| -----: | ----------- |
|      0 | Do nothing  |
|      1 | Flap upward |

---

##  Q-Table Structure

The Q-table is a 4-D tensor:

```
Q[dy_bin, dx_bin, vel_bin, action]
```

Shape:

```
(20, 20, 10, 2)
```

Each entry stores the expected discounted return for taking a given action in a given discretized state.

---

##  Reward Function

The reward design balances **dense shaping** with **strong penalties**:

| Event                        | Reward |
| ---------------------------- | ------ |
| Survival (each step)         | +0.5   |
| Successfully passing a pipe  | +50    |
| Collision (pipe or boundary) | −100   |

This structure:

* Encourages staying alive
* Strongly rewards progress
* Heavily penalizes failure

---

##  Learning Algorithm: Tabular Q-Learning

The agent updates its Q-values using the standard Q-learning update rule:

```
Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]
```

### Hyperparameters

| Parameter | Meaning                 |
| --------- | ----------------------- |
| α         | Learning rate           |
| γ         | Discount factor         |
| ε         | Exploration probability |

---

##  Exploration Strategy (Epsilon-Greedy)

The agent follows an **epsilon-greedy** strategy:

* With probability ε → random action
* With probability 1−ε → greedy action

Epsilon decays over time:

```
ε_start = 1.0
ε_min   = 0.05
ε_decay = 0.995
```

This allows:

* Broad exploration early in training
* Stable exploitation later

---

##  Training Loop

For each episode:

1. Reset environment
2. Observe discretized state
3. Select action using epsilon-greedy
4. Execute action
5. Observe reward and next state
6. Update Q-table
7. Decay epsilon

Training statistics (reward and score) are tracked across episodes.

---

## ▶ Policy Evaluation (Gameplay)

After training:

* Epsilon is set to zero (fully greedy)
* The agent plays Flappy Bird using the learned Q-table
* The environment is rendered in real time using Pygame

The trained agent learns:

* When to flap
* How to align with pipe gaps
* How to survive indefinitely under stable conditions

---

##  Model Saving

The learned policy is saved as a NumPy array:

```
flappy_q_table.npy
```

This file fully represents the trained agent and can be reloaded without retraining.

---
