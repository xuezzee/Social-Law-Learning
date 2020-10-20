import torch
import numpy as np
from env2.Escalator import EscalatorEnv, Escalator_original
# from algorithms.QLearning import Agents
from algorithms.QLearningMOVersion import DQNAgents
from algorithms.ActorCritic import ACAgents
import argparse
import time

def get_args():
    parser = argparse.ArgumentParser(description="arguments of the environment")
    parser.add_argument("--agent_num", type=int, default=6)
    parser.add_argument("--length", type=int, default=15)
    parser.add_argument("--busy_num", type=int, default=3)
    parser.add_argument("--init_area", type=int, default=6)
    parser.add_argument("--n_act", type=int, default=3)
    return parser.parse_args()

def get_params():
    parser = argparse.ArgumentParser(description="parameters of agents")
    parser.add_argument("--algorithm", type=str, default='AC')
    parser.add_argument("--epsilon", type=float, default=0.9)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--n_agents", type=int, default=6)
    parser.add_argument("--gamma", type=float, default=0.9)
    return parser.parse_args()

def get_envSetting(params, env):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    params.s_dim = env.obs_space[0]
    params.a_dim = env.act_space
    params.device = device

def main():
    args = get_args()
    params = get_params()
    env = EscalatorEnv(args)
    # env = Escalator_original(args)
    get_envSetting(params, env)
    if params.algorithm == "DQN":
        agents = DQNAgents(params)
        for ep in range(params.epoch):
            time_step = 0
            ep_reward = 0
            state, reward, done = env.reset(epoch=ep)
            while (False in done):
            # while time_step <= 100:
                if ep >= 80:
                    time.sleep(0.15)
                # if time_step%1000 == 0:
                #     env.save_img()
                env.render()
                transition = []
                action = agents.choose_action(state)
                # next_state, reward, done = env.step(action)
                next_state, reward, done = env.testMode(action)
                ep_reward = ep_reward + sum(reward.values())
                # temp_rew = {}
                # for i in range(len(reward.values())):
                #     temp_rew[i] = sum(reward.values())
                for i in range(params.n_agents):
                    transition.append({"state":state[i], "action":action[i], "next_state":next_state[i], "reward":reward[i]})
                agents.store_transitions(transition)
                state = next_state
                agents.learn()
                del transition
                time_step += 1
            env.print_reward(ep, ep_reward)

    elif params.algorithm == "AC":
        agents = ACAgents(params)
        for ep in range(params.epoch):
            time_step = 0
            ep_reward = 0
            state, reward, done = env.reset(epoch=ep)
            while (False in done):
            # while time_step <= 100:
            #     if ep >= 80:
            #         time.sleep(0.15)
                # if time_step%1000 == 0:
                #     env.save_img()
                env.render()
                transition = []
                action = agents.choose_action(state)
                # next_state, reward, done = env.step(action)
                next_state, reward, done = env.testMode(action)
                ep_reward = ep_reward + sum(reward.values())
                for i in range(params.n_agents):
                    transition.append({"state":state[i], "action":action[i], "next_state":next_state[i], "reward":[reward[i]]})
                # agents.store_transitions(transition)
                state = next_state
                agents.learn(transition)
                del transition
                time_step += 1
            env.print_reward(ep, ep_reward)


if __name__ == '__main__':
    main()
