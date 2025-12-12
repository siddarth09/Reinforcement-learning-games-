import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import PandaPPOModelv0
from env_panda_cfg import PandaEnv


# ============================================================
# 1) GAE Advantage Estimation
# ============================================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: [T]
    values : [T+1] (bootstrapped last value at the end)
    dones  : [T]   (True if episode ended; stops bootstrap)
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])  # 0 if done else 1
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae

    returns = adv + values[:-1]
    return adv, returns


# ============================================================
# 2) PPO Trainer
# ============================================================
class PPOTrainer:
    def __init__(self, obs_dim, act_dim, lr=3e-4, clip=0.2, hidden=256,
                 value_coef=0.5, entropy_coef=1e-2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.ac = PandaPPOModelv0(obs_dim, act_dim, hidden).to(self.device)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr)
        self.clip = clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def update(self, obs, actions, logp_old, returns, advantages,
               action_low, action_high, train_epochs=80, batch_size=256):

        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        logp_old = torch.as_tensor(logp_old, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        # treat advantages as constants during update
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()

        a_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        a_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)

        N = len(obs)
        total_pol = total_val = total_ent = 0.0
        count = 0

        for _ in range(train_epochs):
            idx = np.random.permutation(N)

            for start in range(0, N, batch_size):
                batch = idx[start:start + batch_size]

                # Current log-probs / entropy / values under updated policy
                logp, entropy = self.ac.evaluate_actions(
                    obs[batch], actions[batch], a_low, a_high
                )
                values = self.ac.value(obs[batch])

                ratio = torch.exp(logp - logp_old[batch])
                surr1 = ratio * advantages[batch]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages[batch]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (returns[batch] - values).pow(2).mean()
                ent_bonus = entropy.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * ent_bonus

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.opt.step()

                total_pol += policy_loss.item()
                total_val += value_loss.item()
                total_ent += ent_bonus.item()
                count += 1

        return total_pol / max(count, 1), total_val / max(count, 1), total_ent / max(count, 1)


# ============================================================
# 3) Training Loop
# ============================================================
def train(args):
    # ---- Env & Model ----
    env = PandaEnv(
            args.model_path,
            headless=args.headless,
            w_reach=args.w_reach,
            w_grasp=args.w_grasp,
            w_lift=args.w_lift,
            w_transport=args.w_transport,
            w_success=args.w_success,
            w_collision=args.w_collision,
            w_drop=args.w_drop
        )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ppo = PPOTrainer(
        obs_dim, act_dim,
        lr=args.lr, clip=args.clip, hidden=args.hidden,
        value_coef=0.5, entropy_coef=0.01
    )

    # ---- Logging ----
    writer = SummaryWriter(log_dir=args.logdir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_return = -np.inf

    # Action bounds tensors (kept on device)
    a_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=ppo.device)
    a_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=ppo.device)

    for epoch in range(args.epochs):
        obs_list, act_list, logp_list = [], [], []
        rewards, dones, values = [], [], []

        successes = 0
        # reward components, if provided by env
        reward_components = {
                "reach": [],
                "success": [],
                "joint_penalty": [],
                "accel_penalty": [],
                "obstacle_penalty": []
            }

        obs, _ = env.reset()

        for step in range(args.steps_per_epoch):
            # Act
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=ppo.device)
            action, logp, value = ppo.ac.act(obs_tensor, a_low, a_high)

            action_np = action.detach().cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)

            # Store rollout
            obs_list.append(obs)
            act_list.append(action_np)
            logp_list.append(float(logp.item()))
            rewards.append(float(reward))
            done_flag = bool(terminated or truncated)
            dones.append(done_flag)
            values.append(float(value.item()))

            rew_info = info.get("rew", {})
            for k in reward_components:
                if k in rew_info:
                    reward_components[k].append(float(rew_info[k]))

            # Count successes
            if info.get("success", False):
                successes += 1
            obs = next_obs
            if done_flag:
                obs, _ = env.reset()

        # Bootstrap last value
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=ppo.device)
        _, _, v_final = ppo.ac.act(obs_tensor, a_low, a_high)
        values.append(float(v_final.item()))

        # Compute advantages/returns
        advantages, returns = compute_gae(rewards, values, dones)
        # (re-)norm advantages here for logging consistency; update() will also norm defensively
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        pol_loss, val_loss, entropy = ppo.update(
            obs=np.array(obs_list, dtype=np.float32),
            actions=np.array(act_list, dtype=np.float32),
            logp_old=np.array(logp_list, dtype=np.float32),
            returns=np.array(returns, dtype=np.float32),
            advantages=np.array(advantages, dtype=np.float32),
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            train_epochs=args.train_epochs,
            batch_size=args.batch_size
        )

        # ---------- Epoch stats ----------
        ep_return = float(np.sum(rewards))
        success_rate = successes / float(args.steps_per_epoch)

        mean_actions = float(np.mean(np.abs(act_list))) if len(act_list) > 0 else 0.0
        mean_value_pred = float(np.mean(values[:-1])) if len(values) > 1 else 0.0
        value_error = float(np.mean((np.array(returns) - np.array(values[:-1])) ** 2)) \
            if len(values) > 1 else 0.0

        # Reward components (fallback to 0 if empty)
        def comp_mean(key):
            arr = reward_components[key]
            return float(np.mean(arr)) if len(arr) > 0 else 0.0

        r_reach_m = comp_mean("reach")
        r_success_m = comp_mean("success")
        r_joint_m = comp_mean("joint_penalty")
        r_accel_m = comp_mean("accel_penalty")
        r_obst_m = comp_mean("obstacle_penalty")

        # ---------- Console summary ----------
        print(f"""
            ================= Epoch {epoch} =================
            Return:               {ep_return:.2f}
            Success Rate:         {success_rate*100:.2f}%
            Mean |action|:        {mean_actions:.6f}
            Value Pred Mean:      {mean_value_pred:.6f}
            Value MSE:            {value_error:.6f}
            Policy Loss:          {pol_loss:.6f}
            Value Loss:           {val_loss:.6f}
            Entropy:              {entropy:.6f}
            -------------------------------------------------
            Reach Reward avg:      {r_reach_m:.4f}
            Success Reward avg:    {r_success_m:.4f}
            Joint Penalty avg:     {r_joint_m:.4f}
            Accel Penalty avg:     {r_accel_m:.4f}
            Obstacle Penalty avg:  {r_obst_m:.4f}
            =================================================
            """)

        # ---------- TensorBoard ----------
        writer.add_scalar("Return/EpochTotal", ep_return, epoch)
        writer.add_scalar("Success/Rate", success_rate, epoch)
        writer.add_scalar("Action/MeanAbs", mean_actions, epoch)
        writer.add_scalar("Value/PredMean", mean_value_pred, epoch)
        writer.add_scalar("Value/MSE", value_error, epoch)

        writer.add_scalar("Loss/Policy", pol_loss, epoch)
        writer.add_scalar("Loss/Value", val_loss, epoch)
        writer.add_scalar("Loss/Entropy", entropy, epoch)

        writer.add_scalar("Rewards/Reach", r_reach_m, epoch)
        writer.add_scalar("Rewards/Success", r_success_m, epoch)
        writer.add_scalar("Rewards/JointPenalty", r_joint_m, epoch)
        writer.add_scalar("Rewards/AccelPenalty", r_accel_m, epoch)
        writer.add_scalar("Rewards/ObstaclePenalty", r_obst_m, epoch)

        # ---------- Checkpoints ----------
        if epoch % args.save_freq == 0:
            path = f"{args.checkpoint_dir}/ppo_epoch{epoch}.pt"
            torch.save(ppo.ac.state_dict(), path)
            print(f"Checkpoint saved: {path}")

        if ep_return > best_return:
            best_return = ep_return
            best_path = f"{args.checkpoint_dir}/best_model.pt"
            torch.save(ppo.ac.state_dict(), best_path)
            print(f"New BEST model saved: {best_path}")


# ============================================================
# 4) Argument Parser
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str,
                        default="/home/sid/RL/panda_ppo/franka_emika_panda/scene.xml")

    # PPO/NN
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--hidden", type=int, default=256)

    # Rollouts / Updates
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--steps_per_epoch", type=int, default=4096)
    parser.add_argument("--train_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=512)

    # Infra
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--logdir", type=str, default="runs/panda_training")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    # Reward tuning 
    parser.add_argument("--w_reach", type=float, default=1.0)
    parser.add_argument("--w_grasp", type=float, default=1.0)
    parser.add_argument("--w_lift", type=float, default=1.0)
    parser.add_argument("--w_transport", type=float, default=1.0)
    parser.add_argument("--w_success", type=float, default=1.0)
    parser.add_argument("--w_collision", type=float, default=1.0)
    parser.add_argument("--w_drop", type=float, default=1.0)


    args = parser.parse_args()
    train(args)
