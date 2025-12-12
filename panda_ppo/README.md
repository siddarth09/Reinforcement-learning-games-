# Proximal Policy Optimization for Reachability task (partial pick-place operation)

This repository implements a **reinforcement learning pipeline** for training a Franka Emika Panda robot to reach, interact with, and eventually manipulate objects in a MuJoCo simulation.

The project is intentionally built **from first principles**:

* Custom MuJoCo environment
* Explicit observation and action design
* Dense reward shaping
* A clean PPO implementation (actor–critic)

The goal is not just performance, but **understanding and control** over every learning component.

---

## 1. Environment Overview

The environment is implemented as a custom **Gymnasium-compatible MuJoCo environment**, called `PandaEnv`.

The scene consists of:

* A Franka Panda robot mounted on a table
* Multiple colored cubes placed on the table

At the beginning of **each episode**, **one cube is randomly selected** as the target object. The agent must learn to move its end-effector toward the selected cube. This design encourages **generalization across object locations**, rather than memorization of fixed trajectories.

---

## 2. Observation Space (State Representation)

The observation space is a **20-dimensional continuous vector**, combining robot proprioception and task geometry.

### Observation Vector Structure

```
[ 
  q0 ... q6,          # Joint positions (7)
  dq0 ... dq6,        # Joint velocities (7)
  ee_x, ee_y, ee_z,   # End-effector Cartesian position (3)
  cube_x, cube_y, cube_z  # Target cube position (3)
]
```

### Why This Representation?

* **Joint positions and velocities** provide full robot state awareness.
* **End-effector position** allows the policy to reason in task space.
* **Cube position** makes the task goal-conditioned, enabling transfer to unseen object locations.

This structure is deliberately minimal but sufficient for learning reaching and manipulation behaviors.

---

## 3. Action Space (Control Interface)

The action space is **5-dimensional and continuous**, representing Cartesian motion and gripper control.

### Action Vector

```
[
  dx, dy, dz,     # End-effector Cartesian displacement
  dyaw,           # End-effector yaw rotation
  grip_cmd        # Gripper open/close command
]
```

### Action Limits

```
dx, dy, dz  : [-0.03, 0.03] meters per step
dyaw        : [-0.3, 0.3] radians per step
grip_cmd    : [-1.0, 1.0]
```

### Control Details

* Cartesian actions are converted to joint velocities using the **end-effector Jacobian**.
* The gripper is controlled explicitly:

  * Positive grip command closes the gripper
  * Negative grip command opens the gripper

This separation allows the policy to learn arm motion and gripper behavior independently.

---

## 4. Reward Function (Dense Shaping)

The reward function is **dense, smooth, and modular**, designed to guide learning while maintaining stability.

### Reach Reward (Dense)

Encourages the end-effector to move closer to the target cube:

```
reach_reward = 5 * (1 - tanh(distance_to_cube))
```

Why this works:

* Smooth gradients near the goal
* Bounded values
* Avoids instability of inverse-distance rewards

---

### Success Reward (Sparse Bonus)

A large bonus is given when the end-effector reaches the cube:

```
success if distance_to_cube < 0.03 meters
```

This provides a clear signal for task completion without dominating early learning.

---

### Penalty Terms (Stability and Safety)

Several penalties are included to encourage physically reasonable behavior:

* **Joint limit penalty**
  Discourages configurations near joint limits

* **Action smoothness penalty**
  Penalizes large changes between consecutive actions

* **Obstacle / table penalty**
  Prevents the end-effector from moving too close to the table surface

---

### Final Reward Composition

```
total_reward =
    w_reach     * reach_reward
  + w_success   * success_reward
  + w_collision * obstacle_penalty
  + w_drop      * action_penalty
  + w_grasp     * joint_penalty
  + w_lift      * acceleration_penalty
```

Each reward component is logged separately during training for debugging and analysis.

---

## 5. Learning Algorithm: Proximal Policy Optimization (PPO)

The agent is trained using **Proximal Policy Optimization (PPO)**, an on-policy reinforcement learning algorithm well suited for continuous control.

### Why PPO?

* Stable policy updates using clipped objectives
* Robust to reward scaling
* Widely used in robotics and locomotion tasks
* Simpler and more reliable than vanilla policy gradients

---

## 6. Policy Architecture

The policy is implemented as an **actor–critic neural network** with **separate networks** for policy and value estimation.

### Actor (Policy Network)

* Outputs a **Gaussian distribution** over actions
* Uses tanh squashing and affine scaling to respect action bounds
* Log-probabilities are computed correctly using change-of-variables

### Critic (Value Network)

* Predicts the state value
* Used for advantage estimation and policy stabilization

Separating actor and critic prevents gradient interference and improves learning stability.

---

## 7. Training Procedure

* On-policy rollouts collected from the MuJoCo simulation
* Generalized Advantage Estimation (GAE) used to reduce variance
* PPO updates performed over multiple epochs per rollout
* Entropy regularization encourages exploration
* Best-performing models are checkpointed automatically

---

## 8. Evaluation and Generalization

At test time:

* A trained policy is loaded
* The environment is reset with a **new object location**
* The policy receives the new cube position through observations

Because the policy is **goal-conditioned**, it can reach objects placed at **locations never seen during training**, as long as they lie within the workspace.

---

## 9. Future Extensions

Planned extensions include:

* Curriculum learning: reach → grasp → lift → place
* Explicit goal-conditioned placement
* Fine-tuning from pretrained checkpoints
* Migration to `rsl_rl` for large-scale vectorized training
* Sim-to-real experiments

---

