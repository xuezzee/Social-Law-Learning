import numpy as np
import argparse
from BaseEnv import BaseEnv

class EscalatorEnv():
    def __init__(self, args):
        self.baseEnv = BaseEnv(args)
        self.args = args

    def reset(self):
        self.baseEnv.init_agents()
        self.baseEnv.init_escalator()
        state = self.baseEnv.observation
        get_pos = self.baseEnv.get_agent_pos
        get_Astate = self.baseEnv.agent_state
        for i in range(self.args.agent_num):
            print('%d init_pos:'%self.baseEnv.agents[i]['label'], self.baseEnv.agents[i]['init_pos'])

        obs = [np.concatenate((state.reshape(-1), get_pos(index), get_Astate(index))) for index in range(self.args.agent_num)]
        done = self.baseEnv.done_info
        # reward = self.cal_reward()
        reward = [0 for i in range(self.args.agent_num)]  #temporarily

        return obs, reward, done

    def cal_reward(self):
        return self.baseEnv.reward_cal()

    def step(self, actions):
        self.baseEnv.time_step += 1
        self.baseEnv.change_position(actions)
        self.baseEnv.auto_proceed2()
        state = self.baseEnv.observation
        get_pos = self.baseEnv.get_agent_pos
        get_Astate = self.baseEnv.agent_state

        obs = [np.concatenate((state.reshape(-1), get_pos(index), get_Astate(index))) for index in range(self.args.agent_num)]
        done = self.baseEnv.done_info
        reward = self.cal_reward()

        return obs, reward, done

    def render(self):
        self.baseEnv._render()

    @property
    def obs_space(self):
        return (self.args.length * 2 + 2,)

    @property
    def act_space(self):
        return self.args.n_act



def get_args():
    perse = argparse.ArgumentParser(description="arguments of the environment")
    perse.add_argument("--agent_num", type=int, default=5)
    perse.add_argument("--length", type=int, default=20)
    perse.add_argument("--busy_num", type=int, default=2)
    perse.add_argument("--init_area", type=int, default=8)
    perse.add_argument("--n_act", type=int, default=2)
    return perse.parse_args()


if __name__ == '__main__':
    import time
    args = get_args()
    env = EscalatorEnv(args)
    obs, reward, done = env.reset()
    while True:
        action = np.random.choice(a=env.act_space, size=args.agent_num, replace=True, p=None)
        # print(action)
        env.render()
        time.sleep(1)
        obs, reward, done = env.step(action)
        if not (False in done):
            print(reward)
            print('===================================')
            obs, reward, done = env.reset()
