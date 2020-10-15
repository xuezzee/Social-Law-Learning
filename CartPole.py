import gym
import numpy as np
import torch
from algorithms.QLearning import QLearning
from algorithms.ActorCritic import ActorCritic
from algorithms.QLearningMOVersion import DQN
import argparse
from Logger import Logger

logger = Logger('./logs/CartPole/log2')

def get_params():
    parser = argparse.ArgumentParser()
    return parser

env = gym.make('CartPole-v0').unwrapped
obs_space = env.observation_space
n_act = 2
n_obs = 4
# print(obs_space)
# print(dir(env))
params = get_params()
params.s_dim = n_obs
params.a_dim = n_act
params.epsilon = 0.9
params.device = 'cpu'
params.gamma = 0.9
agent = ActorCritic(params)

print(agent)

epoch = 100000

for ep in range(epoch):
    t_r = 0
    # transition = []
    s = env.reset()
    d = False
    while not d:
        transition = []
        env.render()
        act = agent.choose_action(s[np.newaxis,:])
        # print(act)
        # act = agent.choose_action(s)
        s_, r, d, _ = env.step(act)
        if d:
            r = -10
        transition.append({"state": s, "action": act, "next_state": s_, "reward": [r]})
        # agent.store_transition(transition[0])
        # agent.store_transition(s, act, r, s_)
        s = s_
        agent.learn(transition[0])
        # agent.learn()
        del transition
        t_r += r
    print("reward:",t_r)
    logger.scalar_summary("reward", t_r, ep)