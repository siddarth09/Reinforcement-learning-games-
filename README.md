# Reinforcement learning experiments 

This repository contains two reinforcement learning projects:

1. **Value Iteration on a 5×5 Gridworld**
2. **Q-Learning agent that solves a simplified version of Flappy Bird (Pygame)**

Both projects illustrate the fundamentals of dynamic programming and model-free RL.

---
#  **1. Gridworld — Value Iteration**

##  **Environment**

A 5×5 gridworld:

* Agent can move: **Up, Down, Left, Right**
* Hitting a wall keeps the agent in place
* Every move yields a **step reward = –3**
* The terminal state is **(4,4)** (goal)
* Episode ends at the goal

---

##  **Theory: Value Iteration**

Value Iteration computes the **optimal state-value function**:

[
V^*(s) = \max_a \left[ R(s,a) + \gamma V^*(s') \right]
]

It repeatedly updates the values until convergence:

[
V_{k+1}(s) \leftarrow \max_{a} \left( R(s,a) + \gamma V_k(s') \right)
]

Where:

* ( R(s,a) ) = reward of taking action ( a ) in state ( s )
* ( s' ) = next state after action
* ( \gamma ) = discount factor
* Once the values converge, the optimal policy is extracted:

[
\pi^*(s) = \arg\max_a \left( R(s,a) + \gamma V(s') \right)
]

---

##  **Where the math happens in your code**

### **Value Function Update**

```python
values.append(reward + gamma * v[s_next]) 
v_new[s] = max(values)
```

This is exactly:

[
V(s) = \max_a [ R + \gamma V(s') ]
]

### **Policy Extraction**

```python
val = reward + gamma * v[s_next]
policy[s] = arrow[best_action]
```

This is:

[
\pi(s) = \arg\max_a [ R + \gamma V(s') ]
]

---

#  **2. Flappy Bird — Q-Learning (Pygame)**

Your Flappy Bird agent uses **tabular Q-learning** and achieves **superhuman scores (1700–1900 pipes)** via discretized continuous state space.

---

##  **Environment**

State consists of:

| Variable | Meaning                          |
| -------- | -------------------------------- |
| `dy`     | vertical distance to pipe center |
| `dx`     | horizontal distance to pipe      |
| `vel`    | bird vertical velocity           |

This state is **discretized** into bins so you can use a Q-table.

Actions:

* **0 = do nothing**
* **1 = flap**

Rewards:

* +50 for passing a pipe
* +0.5 for surviving
* −100 for crashing

---

#  **Theory: Q-Learning**

Q-learning learns the **optimal action-value function**:

[
Q^*(s,a) = R + \gamma \max_{a'} Q^*(s', a')
]

The update rule in your code is:

```python
self.Q[state][action] += alpha * (target - q_sa)
```

Which corresponds exactly to:

[
Q(s,a) \leftarrow Q(s,a) + \alpha\left( R + \gamma \max_{a'} Q(s',a') - Q(s,a) \right)
]

This is **off-policy TD learning**, meaning it learns the optimal greedy policy even while taking exploratory actions.

---

#  **Where the math happens in your code**

### **Q-learning update**

```python
q_sa = self.Q[state][action]

if done:
    target = reward
else:
    target = reward + self.gamma * np.max(self.Q[next_state])

self.Q[state][action] = q_sa + self.alpha * (target - q_sa)
```

This *is* the core Q-learning equation.

---

### **Epsilon-greedy Action Selection**

```python
if greedy or np.random.rand() > self.epsilon:
    return int(np.argmax(self.Q[state]))
else:
    return random.choice(ACTIONS)
```

This implements:

* With probability (1 - \epsilon): exploit
* With probability (\epsilon): explore

---



---
#  **Main Loop Explanation**

### **Training loop (`train()`)**

```python
state = env.reset()
done = False

while not done:
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    agent.update(state, action, reward, next_state, done)

    state = next_state
```

This maps to RL math:

1. **Agent chooses action**
   [
   a = \epsilon\text{-greedy}(Q(s))
   ]

2. **Environment transitions**
   [
   (s', r) = \text{env}(s,a)
   ]

3. **Q-learning TD update**
   [
   Q(s,a) \leftarrow Q(s,a) + \alpha\left(r + \gamma \max_{a'}Q(s',a') - Q(s,a)\right)
   ]

4. **State moves forward**

---

#  **Running Flappy Bird Training**

```
python flappy_bird_q_learning.py --episodes 80000 --render-training False
```

---

#  **Watch the trained agent**

```
python flappy_bird_q_learning.py --episodes 0
```

---

#  **Saving the Q-table**

Your script automatically saves:

```
flappy_q_table.npy
```

And loads it if needed.

---

# 🏁 **Conclusion**

This repository teaches:

* **Dynamic programming** with value iteration
* **Tabular TD control** (Q-learning)
* **Discretization of continuous environments**
* **Building and training RL agents in Pygame**
* **Achieving superhuman performance with simple RL algorithms**

Your Flappy Bird agent achieving **2700–2900 pipes** is an exceptional example of how powerful even simple RL can be with the right environment tuning.
