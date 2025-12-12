#  Proximal Policy Optimization (PPO) for Continuous Control of cart pole

## Inverted Pendulum / CartPole (MuJoCo)

This repository implements **Proximal Policy Optimization (PPO)** from scratch for the **InvertedPendulum-v5** MuJoCo environment using Gymnasium.

The purpose of this project is twofold:

1. Build a **clean, minimal PPO baseline** for continuous control
2. Establish a **conceptual and implementation bridge** toward more complex robotic systems (e.g. Franka Panda manipulation)

This implementation intentionally avoids high-level RL libraries in favor of **explicit control over policy design, advantage estimation, and training dynamics**.

<video width="640" height="360" controls autoplay loop muted>
  <source src="../assets/cartpole.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## 1. Environment Overview

The environment used is:

```
InvertedPendulum-v5 (MuJoCo)
```

This task requires stabilizing an inverted pendulum by applying continuous forces at its base. The episode ends when the pole falls beyond a threshold or after a maximum time horizon.

This environment is ideal for PPO because:

* The action space is continuous
* The dynamics are fast and unstable
* The reward signal is dense and well-shaped

---

## 2. Observation Space (State Representation)

The observation space is a **continuous vector** provided directly by the environment.

Typical components include:

* Cart position and velocity
* Pole angle and angular velocity

The policy receives the **full physical state** of the system at each timestep.

---

## 3. Action Space (Continuous Control)

The action space is **one-dimensional and continuous**:

```
a ∈ [-act_limit, act_limit]
```

Where:

* `act_limit` is defined by the environment
* Actions correspond to horizontal forces applied to the cart

### Action Squashing

The policy outputs unconstrained values which are then passed through:

```
tanh(action) * act_limit
```

This ensures:

* Actions always respect environment bounds
* Gradients remain smooth during learning

---

## 4. Policy Architecture (Actor–Critic)

The model is implemented as a **shared Actor–Critic network**, with separate heads.

### Actor Network

* Outputs the mean of a Gaussian policy
* Uses a **state-independent log standard deviation**
* Samples actions from a Normal distribution

Architecture:

* Two hidden layers
* Tanh activations
* Linear output layer for action mean

### Critic Network

* Predicts the state value V(s)
* Same depth and width as the actor
* Separate parameters to avoid gradient interference

This separation stabilizes training and improves value estimation accuracy.

---

## 5. Stochastic Policy and Exploration

The policy is defined as:

```
π(a | s) = Normal(μ(s), σ)
```

Key design choices:

* Gaussian policy enables smooth exploration
* Fixed log standard deviation simplifies training
* Entropy is tracked to monitor exploration behavior

At test time, the policy can be run in:

* **Stochastic mode** (sampling actions)
* **Deterministic mode** (using the mean action)

---

## 6. Advantage Estimation (GAE)

Generalized Advantage Estimation (GAE) is used to reduce variance while maintaining low bias.

Steps:

1. Collect rewards, values, and termination flags
2. Compute temporal-difference errors
3. Accumulate discounted advantages backward in time
4. Normalize advantages before PPO updates

This is critical for stable PPO performance.

---

## 7. PPO Objective

The PPO update uses the **clipped surrogate objective**, which limits how much the policy can change per update.

Core ideas:

* Prevent destructive policy updates
* Maintain monotonic improvement
* Balance exploration and exploitation

Loss components:

* Policy loss (clipped objective)
* Value loss (mean squared error)
* Gradient norm clipping for stability

---

## 8. Training Loop

Training proceeds in repeated cycles of:

1. Collecting on-policy rollouts
2. Computing advantages and returns
3. Performing multiple PPO epochs on the same data
4. Logging metrics and saving checkpoints

Key training features:

* Batched PPO updates
* Advantage normalization
* Early stopping based on reward threshold
* Periodic checkpoint saving

---

## 9. Evaluation and Visualization

A separate test script allows:

* Loading trained models
* Running policies in real-time MuJoCo visualization
* Switching between stochastic and deterministic execution

This enables direct qualitative evaluation of learned behavior.

---

## 10. Why This Baseline Matters

This CartPole / InvertedPendulum PPO implementation serves as:

* A **reference PPO implementation**
* A debugging and validation tool
* A stepping stone toward higher-dimensional robotic control
* A conceptual foundation for the Panda manipulation system

Many architectural and algorithmic ideas here directly extend to:

* Multi-DOF manipulators
* Cartesian control
* Goal-conditioned policies

