# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    """
    PPO Actor–Critic for InvertedPendulum-v5
    Action range = [-3, 3]
    """

    def __init__(self, obs_dim=4, act_dim=1, hidden=64, act_limit=3.0):
        super().__init__()

        self.act_limit = act_limit

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim)
        )

        # Log standard deviation (learnable)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self):
        raise NotImplementedError("Use act(), evaluate_actions(), or value()")

    def act(self, obs):
        """
        Sample action (stochastic during training)
        obs: torch tensor [batch, obs_dim]
        """
        mu = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        logp = dist.log_prob(a).sum(axis=-1)

        # Scale from [-1,1] → [-3,3]
        a = torch.tanh(a) * self.act_limit
        return a, logp

    def evaluate_actions(self, obs, act_unscaled):
        """
        Evaluate log-probability of given actions (for PPO loss)
        UNscaled actions must be passed in BEFORE tanh squashing.
        """
        mu = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)

        logp = dist.log_prob(act_unscaled).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)

        return logp, entropy

    def value(self, obs):
        return self.critic(obs)
