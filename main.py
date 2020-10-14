import torch
import numpy as np
from env2.Escalator import EscalatorEnv
from algorithms import QLearning
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="arguments of the environment")
    parser.add_argument("--agent_num", type=int, default=5)
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--busy_num", type=int, default=2)
    parser.add_argument("--init_area", type=int, default=8)
    parser.add_argument("--n_act", type=int, default=2)
    return parser.parse_args()

def get_params():
    parser = argparse.ArgumentParser(description="parameters of agents")
    parser.add_argument("--n_agent", type=int, default=5)
    parser.add_argument("--s_dim", type=int, default=23)
    parser.add_argument("--a_dim", type=int, default=2)
    parser.add_argument("")


if __name__ == '__main__':
    env =