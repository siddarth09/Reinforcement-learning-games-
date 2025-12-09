# test.py
import gymnasium as gym
import torch
import argparse
from model import ActorCritic
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(args):

    # Create environment
    env = gym.make("InvertedPendulum-v5", render_mode="human")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    # Load model
    model = ActorCritic(obs_dim, act_dim, act_limit=act_limit).to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval()

    print(f"\nLoaded model: {args.model}")
    print(f"Deterministic: {args.deterministic}\n")

    obs, _ = env.reset()
    done = False

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).to(DEVICE).unsqueeze(0)

        with torch.no_grad():

            if args.deterministic:
                # Use mean action (no sampling)
                mu = model.actor(obs_t)
                action = torch.tanh(mu) * model.act_limit
            else:
                action, _ = model.act(obs_t)

        obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
        done = terminated or truncated
        
        time.sleep(args.fps_delay)


# ---------- ARGUMENT PARSER ----------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True,
                        help="Path to .pt checkpoint")

    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic (mean) action instead of sampling")

    parser.add_argument("--fps-delay", type=float, default=0.01,
                        help="Delay between frames for visualization")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test(args)
