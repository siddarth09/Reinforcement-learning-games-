import numpy as np
import torch
import time
import mujoco
import mujoco.viewer

from model import PandaPPOModelv0
from env_panda_cfg import PandaEnv


# -------------------------------------------------------------
# Load Policy
# -------------------------------------------------------------
def load_policy(model_path, obs_dim, act_dim, hidden=512, device="cpu"):
    policy = PandaPPOModelv0(obs_dim, act_dim, hidden).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    return policy


# -------------------------------------------------------------
# Run test with REAL MUJOCO VIEWER
# -------------------------------------------------------------
def run_test(xml_path, model_path, max_steps=1000, hidden=512):

    # Create environment (includes its own model + data)
    env = PandaEnv(xml_path, headless=True)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load policy
    policy = load_policy(model_path, obs_dim, act_dim, hidden, device)

    # Action limits
    a_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
    a_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)

    # Reset environment
    obs, _ = env.reset()

    # Viewer uses SAME model/data from env
    model = env.model
    data = env.data

    mujoco.mj_forward(model, data)

    print("\n🔍 STARTING POLICY TEST WITH VIEWER\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        for step in range(max_steps):

            # Convert observation
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

            # Get PPO action
            with torch.no_grad():
                action, logp, value = policy.act(obs_t, a_low, a_high)

            action_np = action.cpu().numpy()

            # Safety clamp
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

            # ---------- Logging ----------
            print(f"\n--- Step {step} ---")
            print(f"Observation : {np.round(obs[:8], 3)} ...")
            print(f"Action      : {np.round(action_np, 4)}")
            print(f"Log-Prob    : {logp.item():.4f}")
            print(f"Value       : {value.item():.4f}")

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_np)

            print(f"Reward      : {reward:.4f}")
            print(f"Success     : {info.get('success', False)}")

            # Ensure viewer mirrors env state
            mujoco.mj_forward(model, data)
            viewer.sync()

            time.sleep(0.01)  # ~100 FPS

            # Handle termination
            if terminated or truncated:
                print("\n🎉 Episode FINISHED — resetting...\n")
                obs, _ = env.reset()
                mujoco.mj_forward(model, data)

    env.close()


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":
    XML = "/home/sid/RL/panda_ppo/franka_emika_panda/scene.xml"
    MODEL = "checkpoints/best_model.pt"

    run_test(XML, MODEL)
