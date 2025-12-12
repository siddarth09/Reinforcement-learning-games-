import torch
import torch.nn as nn

def train_step(q_net, target_net, buffer, optimizer, device, batch_size=64, gamma=0.99):
    if len(buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    # Q(s, a)
    q_vals = q_net(states)
    q_val = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        target = rewards + gamma * next_q * (1 - dones)

    loss = nn.MSELoss()(q_val, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
