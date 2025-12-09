# train.py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import ActorCritic
import time
import os
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- GAE Advantage Estimation ----------
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae

    returns = adv + values[:-1]
    return adv, returns



# ---------- PPO TRAINER ----------
def train(args):

    env = gym.make("InvertedPendulum-v5")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    model = ActorCritic(obs_dim, act_dim, act_limit=act_limit).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir=args.logdir)

    print("\n=== TRAINING CONFIGURATION ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("===============================\n")

    global_step = 0
    best_mean_reward = -1e9
    patience_counter = 0


    # ---------- PPO LOOP ----------
    for update in range(args.updates):

        obs, _ = env.reset()
        obs_buf, act_unscaled_buf, logp_buf = [], [], []
        rew_buf, val_buf, done_buf, act_buf = [], [], [], []

        steps = 0

        # ---------- COLLECT TRAJECTORY ----------
        while steps < args.steps_per_update:

            obs_t = torch.tensor(obs, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                a, logp = model.act(obs_t.unsqueeze(0))
                v = model.value(obs_t.unsqueeze(0)).item()

            # Invert tanh scaling
            a_unscaled = torch.atanh(a / act_limit)

            next_obs, reward, terminated, truncated, _ = env.step(a.cpu().numpy()[0])
            done = terminated or truncated

            obs_buf.append(obs)
            act_buf.append(a.cpu().numpy()[0])
            act_unscaled_buf.append(a_unscaled.cpu().numpy()[0])
            logp_buf.append(logp.cpu().item())
            rew_buf.append(reward)
            val_buf.append(v)
            done_buf.append(done)

            obs = next_obs
            steps += 1
            global_step += 1

            if done:
                obs, _ = env.reset()

        # ---------- ADVANTAGE COMPUTATION ----------
        with torch.no_grad():
            last_v = model.value(torch.tensor(obs, dtype=torch.float32).to(DEVICE)).item()

        values = np.array(val_buf + [last_v], dtype=np.float32)
        rewards = np.array(rew_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.float32)

        adv, ret = compute_gae(rewards, values, dones, args.gamma, args.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32).to(DEVICE)
        act_unscaled_t = torch.tensor(np.array(act_unscaled_buf), dtype=torch.float32).to(DEVICE)
        logp_old_t = torch.tensor(np.array(logp_buf), dtype=torch.float32).to(DEVICE)
        ret_t = torch.tensor(ret, dtype=torch.float32).to(DEVICE)
        adv_t = torch.tensor(adv, dtype=torch.float32).to(DEVICE)

        # ---------- PPO UPDATE ----------
        for epoch in range(args.ppo_epochs):
            idx = np.arange(len(obs_buf))
            np.random.shuffle(idx)

            for start in range(0, len(idx), args.batch_size):
                batch_idx = idx[start:start + args.batch_size]

                batch_obs = obs_t[batch_idx]
                batch_act_unscaled = act_unscaled_t[batch_idx]
                batch_logp_old = logp_old_t[batch_idx]
                batch_ret = ret_t[batch_idx]
                batch_adv = adv_t[batch_idx]

                logp, entropy = model.evaluate_actions(batch_obs, batch_act_unscaled)
                ratio = torch.exp(logp - batch_logp_old)

                clip_adv = torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio) * batch_adv
                pi_loss = -(torch.min(ratio * batch_adv, clip_adv)).mean()

                v = model.value(batch_obs).squeeze()
                v_loss = ((batch_ret - v) ** 2).mean()

                loss = pi_loss + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()


        # ---------- LOGGING ----------
        mean_reward = float(np.mean(rewards))

        writer.add_scalar("Loss/pi", pi_loss.item(), global_step)
        writer.add_scalar("Loss/v", v_loss.item(), global_step)
        writer.add_scalar("Stats/mean_reward", mean_reward, global_step)

        print(f"Update {update} | reward={mean_reward:.2f} | pi_loss={pi_loss:.3f}")

        # ---------- EARLY STOPPING ----------
        if mean_reward >= args.early_stop_reward:
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter >= args.early_stop_patience:
            print(f"\n EARLY STOPPING at update {update}! Reward={mean_reward:.3f}\n")

            # SAVE FINAL MODEL
            os.makedirs("checkpoints", exist_ok=True)
            final_path = f"checkpoints/ppo_pendulum_final.pt"
            torch.save(model.state_dict(), final_path)
            print(f"💾 Saved final model to: {final_path}\n")

            break


        # ---------- SAVE MODEL ----------
        if update % args.save_every == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/ppo_pendulum_{update}.pt")

    writer.close()




# ---------- ARGUMENT PARSER ----------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--steps-per-update", type=int, default=4000)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=10)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)

    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--logdir", type=str, default="./ppo_logs")

    # NEW ARGUMENTS FOR EARLY STOPPING
    parser.add_argument("--early-stop-reward", type=float, default=0.995)
    parser.add_argument("--early-stop-patience", type=int, default=5)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
