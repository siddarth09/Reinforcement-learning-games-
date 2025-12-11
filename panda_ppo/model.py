import torch
import torch.nn as nn
import torch.nn.functional as F


class PandaPPOModelv0(nn.Module):
    """
    PPO actor-critic with:
      - Separate actor and critic trunks (no gradient entanglement)
      - Gaussian policy + tanh squashing + affine scale to [low, high]
      - Correct log-prob using tanh change-of-variables (Jacobian) correction
    """
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()

        # ----- Actor -----
        self.actor_body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        # state-independent log-std (stable, simple). You can switch to a head if needed.
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # ----- Critic -----
        self.critic_body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.v_head = nn.Linear(hidden, 1)

        self._eps = 1e-6

    # -------- Critic: value only --------
    def value(self, obs):
        z = self.critic_body(obs)
        return self.v_head(z).squeeze(-1)  # [B]

    # -------- Actor helpers --------
    def _actor_dist(self, obs):
        z = self.actor_body(obs)
        mu = self.mu(z)
        std = torch.exp(self.log_std).clamp_min(self._eps)
        return torch.distributions.Normal(mu, std)  # diag Normal

    @staticmethod
    def _squash_and_scale(raw_action, low, high, eps=1e-6):
        # tanh to (-1,1), then affine to [low, high]
        squashed = torch.tanh(raw_action)
        # Ensure bounds are tensors broadcastable to action
        mid = (high + low) / 2.0
        half = (high - low) / 2.0
        action = mid + half * squashed
        # tiny clamp to avoid exact bound issues
        return torch.clamp(action, low + eps, high - eps), squashed, mid, half

    # Change-of-variables: log p(a) from raw pre-tanh Normal
    def _logprob_squashed(self, dist, raw_action, squashed, half):
        # base log prob for raw_action
        base_logp = dist.log_prob(raw_action).sum(-1)  # [B]
        # tanh Jacobian: prod (1 - tanh(x)^2)
        tanh_corr = torch.log(1.0 - squashed.pow(2) + self._eps).sum(-1)
        # scale Jacobian: prod (half_i), i.e., |d a / d squashed| = half
        scale_corr = torch.log(half + self._eps).sum(-1)
        # log p(a) = log p(raw) - [sum log(1 - tanh^2) + sum log(half)]
        return base_logp - tanh_corr - scale_corr

    # -------- Public actor API --------
    def act(self, obs, action_low, action_high):
        """
        Sample action and return:
          action in [low, high], log_prob(action), value(obs)
        """
        dist = self._actor_dist(obs)
        raw_action = dist.rsample()  # reparameterized sample
        action, squashed, mid, half = self._squash_and_scale(raw_action, action_low, action_high, self._eps)
        logp = self._logprob_squashed(dist, raw_action, squashed, half)
        v = self.value(obs)
        return action, logp, v

    def evaluate_actions(self, obs, actions, action_low, action_high):
        """
        For PPO update: get log_prob(actions) and an entropy term.
        We invert the squash+scale to compute correct log-prob.
        """
        # Ensure tensors
        dist = self._actor_dist(obs)

        # Invert affine to (-1,1)
        mid = (action_high + action_low) / 2.0
        half = (action_high - action_low) / 2.0
        y = (actions - mid) / (half + self._eps)  # should be in (-1,1)

        # Clamp to avoid atanh blow-ups
        y = torch.clamp(y, -1.0 + self._eps, 1.0 - self._eps)

        # Invert tanh: atanh(y)
        raw_action = 0.5 * (torch.log1p(y) - torch.log1p(-y))  # atanh(y)

        # Compute log-prob with correction
        logp = self._logprob_squashed(dist, raw_action, y, half)

        # Entropy: use base Normal entropy (common, stable approximation)
        entropy = dist.entropy().sum(-1)

        return logp, entropy
