import numpy as np 
import torch 
import torch.nn as nn
import torch.functional as F 


class PandaPPOModelv0(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden):

        super().__init__() 

        self.arm = nn.Sequential(
            nn.Linear(obs_dim,hidden),
            nn.ReLU(), 
            nn.Linear(hidden,hidden),
            nn.ReLU(),
        )

        self.mu = nn.Linear(hidden,act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim)) 

        # Value head

        self.V = nn.Linear(hidden,1)


    def forward(self,obs):
        z = self.arm(obs)
        mu = self.mu(z) 
        std = torch.exp(self.log_std)
        value = self.V(z)

        return mu,std,value 
    
    def act(self,obs):

        mu,std,value = self.forward(obs)
        dist = torch.distributions.Normal(mu,std)
        action = dist.sample() 
        logp = dist.log_prob(action).sum(axis= -1)
        return action,logp,value 
    
    def critic(self,obs,action):
        mu,std,value= self.forward(obs) 
        dist = torch.distributions.Normal(mu,std)
        logp = dist.log_prob(action).sum(axis =-1)

        entropy = dist.entropy().sum(axis = -1)
        return logp,entropy,value 
    

    
