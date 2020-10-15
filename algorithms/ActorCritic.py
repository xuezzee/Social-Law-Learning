import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim=128, device='cpu'):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.device = device

    def build_network(self):
        self.linear1 = nn.Linear(self.s_dim, self.h_dim)
        self.linear2 = nn.Linear(self.h_dim, self.h_dim)
        self.linear3 = nn.Linear(self.h_dim, self.a_dim)

    def forward(self, input):
        x = self.convert_type(input)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.softmax(self.linear3(x))
        return x

    def convert_type(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = x.to(self.device)
        return x


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim=128, device='cpu'):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.device = device

    def build_network(self):
        self.linear1 = nn.Linear(self.s_dim, self.h_dim)
        self.linear2 = nn.Linear(self.h_dim, self.h_dim)
        self.linear3 = nn.Linear(self.h_dim, 1)

    def forward(self, input):
        x = self.convert_type(input)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def convert_type(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = x.to(self.device)
        return x

class ActorCritic():
    def __init__(self, params):
        self.params = params
        self.actor = Actor(params.s_dim, params.a_dim, device=params.device)
        self.critic = Critic(params.s_dim, params.a_dim, device=params.device)
        self.optim = {'optA':torch.optim.Adam(self.actor.parameters(), lr=0.001),
                      'optC':torch.optim.Adam(self.actor.parameters(), lr=0.001)}
        self.lr_scheduler = {'lrA':torch.optim.lr_scheduler.StepLR(self.optim['optA'], 1000, 0.9, last_epoch=-1),
                             'lrC':torch.optim.lr_scheduler.StepLR(self.optim['optC'], 1000, 0.9, last_epoch=-1)}

    def choose_action(self, state):
        prob = self.actor(state)

