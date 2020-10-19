import numpy as np
import argparse
from env2.BaseEnv import BaseEnv, Base_original
from Logger import Logger

logger = Logger('./log/logDQNLearn2')

class EscalatorEnv():
    def __init__(self, args):
        self.baseEnv = BaseEnv(args)
        self.args = args

    def reset(self, epoch=None):
        self.baseEnv.init_agents()
        self.baseEnv.init_escalator()
        state = self.baseEnv.observation
        get_pos = self.baseEnv.get_agent_pos
        get_Astate = self.baseEnv.agent_state
        self.baseEnv.info(epoch=epoch)
        for i in range(self.args.agent_num):
            print('%d init_pos:'%self.baseEnv.agents[i]['label'], self.baseEnv.agents[i]['init_pos'])
        obs = [np.concatenate((state.reshape(-1), get_pos(index).reshape(-1), get_Astate(index))) for index in range(self.args.agent_num)]
        done = self.baseEnv.done_info
        # reward = self.cal_reward()
        reward = [0 for i in range(self.args.agent_num)]  #temporarily

        return obs, reward, done

    def cal_reward(self):
        return self.baseEnv._reward_cal2()

    def step(self, actions):
        self.baseEnv.time_step += 1
        self.baseEnv.change_position(actions)
        self.baseEnv.auto_proceed2()
        state = self.baseEnv.observation
        get_pos = self.baseEnv.get_agent_pos
        get_Astate = self.baseEnv.agent_state

        obs = [np.concatenate((state.reshape(-1), get_pos(index).reshape(-1), get_Astate(index))) for index in range(self.args.agent_num)]
        done = self.baseEnv.done_info
        reward = self.cal_reward()

        return obs, reward, done

    def render(self):
        self.baseEnv._render()

    def testMode(self, actions):
        idle_agents = self.baseEnv.get_idle_agents()
        for i in idle_agents:
            actions[i] = 1
        return self.step(actions)

    @property
    def obs_space(self):
        return ((self.args.length+1) * 4 + 1,)

    @property
    def act_space(self):
        return self.args.n_act

    def print_reward(self, epoch, reward=None):
        if reward != None:
            logger.scalar_summary("total reward", reward, epoch)
            print("ep_reward:",reward)
        else:
            reward = self.baseEnv.rewardRec
            print(end='\n')
            for i in range(self.args.agent_num):
                print("agent{label}:{reward}|".format(label=i, reward=reward[i]),end='')
            print(end='\n')
            print("total reward:", sum(reward.values()),end='\n')
            logger.scalar_summary("total reward", sum(reward.values()), epoch)


'''
-------------------------------------------------------
The original version of escalator environment
-------------------------------------------------------
'''

class Escalator_original(EscalatorEnv):
    def __init__(self, args):
        super().__init__(args)
        self.baseEnv = Base_original(args)

    def cal_reward(self):
        return self.baseEnv.reward_cal()

    def step(self, actions):
        self.baseEnv.time_step += 1
        self.baseEnv.change_position(actions)
        self.baseEnv.auto_proceed()
        state = self.baseEnv.observation
        get_pos = self.baseEnv.get_agent_pos
        get_Astate = self.baseEnv.agent_state

        obs = [np.concatenate((state.reshape(-1), get_pos(index).reshape(-1), get_Astate(index))) for index in range(self.args.agent_num)]
        done = self.baseEnv.done_info
        reward = self.cal_reward()

        return obs, reward, done


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
        time.sleep(0.1)
        obs, reward, done = env.step(action)
        if not (False in done):
            # print('reward',reward)
            # print('========================================')
            env.print_reward()
            obs, reward, done = env.reset()
