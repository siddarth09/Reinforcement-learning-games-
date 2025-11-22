Here is your updated README with **all equations converted to proper LaTeX equation environments using `\begin{equation}` … `\end{equation}`**, plus all formatting preserved and improved.

---

# 🚀 Reinforcement Learning Experiments

This repository contains multiple reinforcement learning projects implementing both **dynamic programming** and **model-free RL** techniques using simple environments and custom games.

Current projects included:

1. **Value Iteration on a 5×5 Gridworld**
2. **Flappy Bird — Q-Learning (Pygame)**
3. **Taxi Driver — Deep Q-Network (DQN)** ← *new*

---

# 1️⃣ Gridworld — Value Iteration

## Environment

A 5×5 grid:

* Actions: Up, Down, Left, Right
* Bumping into walls leaves the agent in place
* Step reward = **–3**
* Terminal state = **(4,4)**
* The optimal policy is computed analytically using **value iteration**

---

## Theory: Value Iteration

Value iteration computes the optimal value function:

\begin{equation}
V^*(s) = \max_a \left[ R(s,a) + \gamma V^*(s') \right]
\end{equation}

It repeatedly applies:

\begin{equation}
V_{k+1}(s) = \max_{a} \left( R(s,a) + \gamma V_k(s') \right)
\end{equation}

After convergence, the optimal policy is:

\begin{equation}
\pi^*(s) = \arg\max_a \left( R(s,a) + \gamma V(s') \right)
\end{equation}

---

## Where the math happens in your code

### Value Update (matches the Bellman optimality update)

```python
values.append(reward + gamma * v[s_next])
v_new[s] = max(values)
```

This corresponds to:

\begin{equation}
V(s) = \max_a \left[ R + \gamma V(s') \right]
\end{equation}

---

### Policy Extraction

```python
policy[s] = arrow[best_action]
```

Which corresponds to:

\begin{equation}
\pi(s) = \arg\max_a \left[ R + \gamma V(s') \right]
\end{equation}

---

# 2️⃣ Flappy Bird — Q-Learning (Pygame)

A simplified Flappy Bird environment built in **Pygame**, using tabular Q-learning with discretized continuous states.

State includes:

| Variable | Meaning                          |
| -------- | -------------------------------- |
| `dy`     | vertical distance to pipe center |
| `dx`     | horizontal distance to pipe      |
| `vel`    | bird vertical velocity           |

Actions:

* **0 = no flap**
* **1 = flap**

Rewards:

* `+50` for passing a pipe
* `+0.5` for surviving
* `–100` for crashing

---

# Theory: Q-Learning

Q-learning learns the optimal action-value function:

\begin{equation}
Q^*(s,a) = R + \gamma \max_{a'} Q^*(s', a')
\end{equation}

The update rule:

\begin{equation}
Q(s,a) \leftarrow
Q(s,a) +
\alpha \Big( R + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big)
\end{equation}

Your implementation uses:

```python
target = reward + self.gamma * np.max(self.Q[next_state])
self.Q[state][action] += self.alpha * (target - q_sa)
```

---

### Epsilon-greedy Action Selection

\begin{equation}
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \
\arg\max_a Q(s,a), & \text{with probability } 1 - \epsilon
\end{cases}
\end{equation}

Training:

```
python flappy_bird_q_learning.py --episodes 80000
```

Testing:

```
python flappy_bird_q_learning.py --episodes 0
```

Best recorded score: **4972 pipes**

---

# 3️⃣ Taxi Driver — Deep Q-Network (DQN) 🚕

## Overview

This project trains a DQN agent on the **Taxi-v3** environment from Gymnasium.

Taxi-v3:

* 500 discrete states
* 6 actions
* Rewards for correct pickup/dropoff
* Penalties for illegal moves and step cost
* Perfect environment for discrete DQN

---

# Theory: Deep Q-Network (DQN)

DQN approximates the Q-function using a neural network:

\begin{equation}
Q(s,a;\theta) \approx Q^*(s,a)
\end{equation}

Instead of storing a table, the network outputs Q-values for all actions.

The TD target is:

\begin{equation}
y = r + \gamma \max_{a'} Q(s', a'; \theta^{-})
\end{equation}

And the loss minimized is:

\begin{equation}
L = \big( Q(s,a; \theta) - y \big)^2
\end{equation}

DQN uses two critical components:

1. **Experience Replay**
2. **Target Network**

These stabilize learning significantly.

---

# 🔧 Training the Taxi Driver Agent

Use:

```
python taxi_driver_train.py
```

This trains a DQN with:

* One-hot encoded 500-dimensional states
* Experience replay
* Target network updates
* Epsilon-greedy exploration
* Adam optimizer

The trained model is saved as:

```
taxi_dqn.pth
```

---

# 🎮 Testing the Trained Taxi Policy

Run:

```
python test_taxi_rl_driver.py
```

This will:

* Load `taxi_dqn.pth`
* Render the Taxi-v3 grid
* Execute the learned optimal policy

The agent performs pickup & dropoff efficiently with minimal penalty.

---




