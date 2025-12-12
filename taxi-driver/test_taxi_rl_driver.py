import gymnasium as gym
import torch
import numpy as np
from q_learning_mlp import QNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make("Taxi-v3", render_mode="human")

state_dim = 500
action_dim = env.action_space.n

def one_hot(s):
    v = np.zeros(state_dim)
    v[s] = 1.0
    return v

policy = QNetwork(state_dim, action_dim).to(device)
policy.load_state_dict(torch.load("taxi_dqn.pth", map_location=device))
policy.eval()

state, _ = env.reset()
state = one_hot(state)

while True:
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32).to(device)
        action = policy(s).argmax().item()

    next_state, reward, done, trunc, _ = env.step(action)
    state = one_hot(next_state)

    if done:
        state, _ = env.reset()
        state = one_hot(state)
