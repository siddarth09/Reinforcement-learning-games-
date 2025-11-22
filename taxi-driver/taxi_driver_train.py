import gymnasium as gym
import numpy as np
import torch
from q_learning_mlp import QNetwork
from replay_buffer import ReplayBuffer
from dqn_train_setup import train_step
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

env = gym.make("Taxi-v3")

def get_args():
    parser= argparse.ArgumentParser("Taxi Driver Helper")
    parser.add_argument("--episodes", type=int, default=3000,
                        help="Number of training episodes")
    
    return parser.parse_args()

state_dim = 500  # one-hot states
action_dim = env.action_space.n

def one_hot(s):
    v = np.zeros(state_dim)
    v[s] = 1.0
    return v

q_net = QNetwork(state_dim, action_dim).to(device)
target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(q_net.state_dict())

optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
buffer = ReplayBuffer()

episodes = 5000
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.999

for ep in range(episodes):
    state, _ = env.reset()
    state = one_hot(state)
    total_reward = 0

    done = False
    while not done:
        # Epsilon greedy algo
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).to(device)
                action = q_net(s).argmax().item()

        next_state, reward, done, trunc, _ = env.step(action)
        next_state_oh = one_hot(next_state)

        buffer.add((state, action, reward, next_state_oh, done))
        state = next_state_oh
        total_reward += reward

        train_step(q_net, target_net, buffer, optimizer, device)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if ep % 50 == 0:
        target_net.load_state_dict(q_net.state_dict())

    print(f"Ep {ep} | Reward: {total_reward} | Epsilon: {epsilon:.3f}")

torch.save(q_net.state_dict(), "taxi_dqn.pth")
