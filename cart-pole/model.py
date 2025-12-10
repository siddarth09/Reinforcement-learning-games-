import torch 
import torch.nn as nn 
import torch.nn.functional as F 
 

class ActorCritic(nn.Module):
    """
    PPO Actor-Critic for inverted pendulum 
    """
    def __init__(self,obs_dim,act_dim,hidden,act_limit):
        super().__init__() 
        self.act_limit=act_limit 

        # Actor network 
        self.actor = nn.Sequential(
            nn.Linear(obs_dim,hidden),
            nn.Tanh(),
            nn.Linear(hidden,hidden),
            nn.Tanh(),
            nn.Linear(hidden,act_dim)
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim,hidden),
            nn.Tanh(),
            nn.Linear(hidden,hidden),
            nn.Tanh(),
            nn.Linear(hidden,1)
        )


    def forward(self):
        raise NotImplementedError("Use act(), evaluate_actions(), or value()")
    
    def act(self,obs):
        mu = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu,std)

        a = dist.sample()
        logp= dist.log_prob(a).sum(axis=-1)

        a = torch.tanh(a)*self.act_limit 

        return a,logp
    
    def evaluate_actions(self,obs,act_unscaled):
        mu = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu,std) 
        log_p = dist.log_prob(act_unscaled).sum(axis = -1)

        entropy = dist.entropy().sum(axis=-1)

        return log_p,entropy
    

    def value(self,obs):
        return self.critic(obs) 
    