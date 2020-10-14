import torch
import numpy as np
from torch import nn
import random
import torch.functional as F
from collections import namedtuple


class ReplayBuffer():
    '''
    Replay Buffer: To store transitions sampled from the interactions with the env.
    '''
    def __init__(self, capacity, params):
        self.capacity = capacity
        self.memory = {'state':np.empty([self.capacity, params.s_dim]),
                       'action':np.empty([self.capacity, 1]),
                       'next_state':np.empty([self.capacity, params.s_dim]),
                       'reward':np.empty([self.capacity, 1])}
        self.pointer = 0

    def push(self, transition):
        '''
        The method used to store trnasitions
        :param transition: a dict, with four transition elements.
        :return: None
        '''
        if self.pointer == self.capacity:
            self.memory['state'] = np.roll(self.memory['state'], shift=1, axis=0)
            self.memory['action'] = np.roll(self.memory['action'], shift=1, axis=0)
            self.memory['next_state'] = np.roll(self.memory['next_state'], shift=1, axis=0)
            self.memory['reward'] = np.roll(self.memory['reward'], shift=1, axis=0)
            self.pointer -= 1

        self.memory['state'][range(self.pointer, self.pointer+1)] = np.array(transition['state'])
        self.memory['action'][range(self.pointer, self.pointer+1)] = np.array(transition['action'])
        self.memory['next_state'][range(self.pointer, self.pointer+1)] = np.array(transition['next_state'])
        self.memory['reward'][range(self.pointer, self.pointer+1)] = np.array(transition['reward'])
        self.pointer += 1

    def sample(self, batch_size):
        '''
        The function to sample transition samples which is going to be used to train the network.
        :param batch_size: how much samples of transitions you want to get
        :return: a sample of transitions
        '''
        index = np.random.choice(a=self.pointer, size=batch_size, replace=False, p=None)
        sample = {'state':self.memory['state'][index],
                  'action':self.memory['action'][index],
                  'next_state':self.memory['next_state'][index],
                  'reward':self.memory['reward'][index]}
        return sample



class Q_Net(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim=128, device='cpu', epsilon=0.9):
        '''
        To build a Q-Net used for individual Q-Learning agent
        :param s_dim: the dimension of state space (int)
        :param a_dim: the dimension of action space (int)
        :param h_dim: the number of hidden neurons (int)[optional]
        '''
        super(Q_Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.device = device
        self.build_network()
        self.epsilon = epsilon

    def build_network(self):
        self.linear1 = nn.Linear(self.s_dim, self.h_dim)
        self.linear2 = nn.Linear(self.h_dim, self.h_dim)
        self.linear3 = nn.Linear(self.h_dim, self.a_dim)

    def forward(self, input):
        x = self.convert_type(input)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))

        return x

    def convert_type(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device)
        return x



class QLearning():
    """Q-Learning algorithm"""

    def __init__(self, params):
        self.params = params
        self.epsilon = params.epsilon
        self.policy_net = Q_Net(params.s_dim, params.a_dim, device=params.device, epsilon=params.epsilon).to(params.device)
        self.target_net = Q_Net(params.s_dim, params.a_dim, device=params.device, epsilon=params.epsilon).to(params.device)
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.scheduler_lr = torch.optim.lr_scheduler.StepLR(self.optim, 1000, gamma=0.9, last_epoch=-1)
        self.replay_buffer = ReplayBuffer(1000, params)

    def choose_action(self, state):
        q_value = self.policy_net(state)
        action = torch.argmax(q_value, dim=-1)
        rand = random.random()
        if rand < self.epsilon:
            return action.cpu().detach().numpy()[0]
        else:
            action = np.random.choice(a=self.params.a_dim, size=1, replace=False, p=None)
            return action[0]

    def store_transition(self, transition):
        self.replay_buffer.push(transition)

    def learn(self):
        if self.replay_buffer.pointer > 10:
            sample = self.replay_buffer.sample(10)
            q_value = self.policy_net(sample['state'])
            q_value = torch.max(q_value, dim=-1)[0]
            q_value_ = self.target_net(sample['next_state'])
            q_value_ = torch.max(q_value_, dim=-1)[0]
            error = q_value - (q_value_ + torch.Tensor(sample['reward']).T)
            loss = error.square().mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler_lr.step()
            print("loss:",loss)

        self.target_net.load_state_dict(self.policy_net.state_dict())


class Agents():
    def __init__(self, params):
        self.params = params
        self.agents = [QLearning(params) for i in range(params.n_agents)]

    def choose_action(self, state):
        action = []
        for i in range(self.params.n_agents):
            agent = self.agents[i]
            act = agent.choose_action(torch.Tensor(state[i]).unsqueeze(0))
            action.append(act)

        return action

    def learn(self):
        for agent in self.agents:
            agent.learn()

    def store_transitions(self, transitions):
        for i in range(self.params.n_agents):
            self.agents[i].store_transition(transitions[i])