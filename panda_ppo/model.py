import torch
import torch.nn as nn
import torch.nn.functional as F


class PandaPPOModelv0(nn.Module):
    """
    PPO actor-critic with:
      - Deep actor trunk (1024 → 512 → 256 → 128)
      - Critic trunk separate
      - Gaussian policy + tanh squashing + affine scaling
      - Correct tanh change-of-variables log-prob correction
    """

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()

        # ---------------- ACTOR NETWORK ----------------
        self.actor_body = nn.Sequential(
            nn.Linear(obs_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )

        # The LAST OUTPUT SIZE of actor_body is 128
        self.mu = nn.Linear(128, act_dim)

        # learnable log_std
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # ---------------- CRITIC NETWORK ----------------
        self.critic_body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.v_head = nn.Linear(hidden, 1)

        self._eps = 1e-6

    # ---------------- VALUE FUNCTION ----------------
    def value(self, obs):
        z = self.critic_body(obs)
        return self.v_head(z).squeeze(-1)

    # ---------------- ACTOR FORWARD ----------------
    def _actor_dist(self, obs):
        z = self.actor_body(obs)
        mu = self.mu(z)
        std = torch.exp(self.log_std).clamp_min(self._eps)
        return torch.distributions.Normal(mu, std)

    @staticmethod
    def _squash_and_scale(raw_action, low, high, eps=1e-6):
        squashed = torch.tanh(raw_action)

        mid = (high + low) / 2.0
        half = (high - low) / 2.0

        action = mid + half * squashed
        action = torch.clamp(action, low + eps, high - eps)
        return action, squashed, mid, half

    def _logprob_squashed(self, dist, raw_action, squashed, half):
        base_logp = dist.log_prob(raw_action).sum(-1)

        tanh_corr = torch.log(1 - squashed.pow(2) + self._eps).sum(-1)
        scale_corr = torch.log(half + self._eps).sum(-1)

        return base_logp - tanh_corr - scale_corr

    def act(self, obs, action_low, action_high):
        dist = self._actor_dist(obs)
        raw_action = dist.rsample()

        action, squashed, mid, half = self._squash_and_scale(
            raw_action, action_low, action_high, self._eps
        )

        logp = self._logprob_squashed(dist, raw_action, squashed, half)
        v = self.value(obs)
        return action, logp, v

    def evaluate_actions(self, obs, actions, action_low, action_high):
        dist = self._actor_dist(obs)

        mid = (action_high + action_low) / 2.0
        half = (action_high - action_low) / 2.0

        y = (actions - mid) / (half + self._eps)
        y = torch.clamp(y, -1 + self._eps, 1 - self._eps)

        raw_action = 0.5 * (torch.log1p(y) - torch.log1p(-y))  # atanh(y)

        logp = self._logprob_squashed(dist, raw_action, y, half)
        entropy = dist.entropy().sum(-1)

        return logp, entropy
