import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim=20, device='cpu'):
        super(Actor, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.device = device
        self.build_network()

    def build_network(self):
        self.linear1 = nn.Linear(self.s_dim, self.h_dim)
        # self.linear2 = nn.Linear(self.h_dim, self.h_dim)
        self.linear3 = nn.Linear(self.h_dim, self.a_dim)

    def forward(self, input):
        x = self.convert_type(input)
        x = torch.relu(self.linear1(x))
        # x = torch.relu(self.linear2(x))
        x = torch.softmax(self.linear3(x), dim=-1)
        return x

    def convert_type(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = x.to(self.device)
        return x


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim=20, device='cpu'):
        super(Critic, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.device = device
        self.build_network()

    def build_network(self):
        self.linear1 = nn.Linear(self.s_dim, self.h_dim)
        # self.linear2 = nn.Linear(self.h_dim, self.h_dim)
        self.linear3 = nn.Linear(self.h_dim, 1)

    def forward(self, input):
        x = self.convert_type(input)
        x = torch.relu(self.linear1(x))
        # x = torch.relu(self.linear2(x))
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
        self.device = params.device
        self.actor = Actor(params.s_dim, params.a_dim, device=params.device).to(self.device)
        self.critic = Critic(params.s_dim, params.a_dim, device=params.device).to(self.device)
        self.optim = {'optA':torch.optim.Adam(self.actor.parameters(), lr=0.001),
                      'optC':torch.optim.Adam(self.critic.parameters(), lr=0.01)}
        self.lr_scheduler = {'lrA':torch.optim.lr_scheduler.StepLR(self.optim['optA'], 1000, 1, last_epoch=-1),
                             'lrC':torch.optim.lr_scheduler.StepLR(self.optim['optC'], 1000, 1, last_epoch=-1)}

    def choose_action(self, state):
        self.prob = self.actor(state)
        m = Categorical(self.prob)
        action = m.sample()
        return int(action.data.cpu().numpy())

    def cal_td(self, transition):
        s = torch.Tensor(transition['state']).to(self.device)
        s_ = torch.Tensor(transition['next_state']).to(self.device)
        r = torch.Tensor(transition['reward']).to(self.device)
        q = self.critic(s)
        q_ = self.critic(s_)
        td_err = q - r - self.params.gamma * q_.detach()
        return td_err

    def learnActor(self, transition, td_err):
        td_err = td_err.detach()
        log_prob = torch.log(self.prob[0][transition['action']])
        loss = (log_prob * td_err)
        self.optim["optA"].zero_grad()
        loss.backward()
        self.optim["optA"].step()
        self.lr_scheduler["lrA"].step()

    def learnCritic(self, transition):
        td_err = self.cal_td(transition)
        loss = td_err.pow(2)
        self.optim['optC'].zero_grad()
        loss.backward()
        self.optim['optC'].step()
        self.lr_scheduler['lrC'].step()
        return td_err

    def learn(self, transition):
        td_err = self.learnCritic(transition)
        self.learnActor(transition, td_err)


class ACAgents():
    def __init__(self, params):
        self.params = params
        self.device = params
        self.agents = [ActorCritic(params) for i in range(params.n_agents)]

    def choose_action(self, state):
        actions = []
        for i in range(self.params.n_agents):
            agent = self.agents[i]
            actions.append(agent.choose_action(state[i]))

        return actions

    def learn(self, transition):
        for i in range(self.params.n_agents):
            agent = self.agents[i]
            agent.learn(transition=transition[i])


