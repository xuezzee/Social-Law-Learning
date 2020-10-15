import numpy as np
import argparse
import tkinter
import copy

ACTIONS = {0:"left",
           1:"right",
           2:"proceed",
           3:"stay"}

class BaseEnv():
    def __init__(self, args):
        '''
        initiative function
        :param args: a dict including:
                        agent_num (int)
                        length (int)
                        busy_num (int)
                        init_area (int)[the length of the area where agents can initate, smaller than the length]
        '''
        self.agent_num = args.agent_num
        self.length = args.length + 1
        self.busy_num = args.busy_num
        self.init_area = args.init_area
        assert self.init_area > self.agent_num, "the initative area is not enough for all agents"
        self.agents = [{} for i in range(self.agent_num)]
        self.root = None

    def init_escalator(self):
        self.escalator = np.full((self.length, 2), "empty***")
        shapeEsc = self.escalator.shape
        self.escalator = self.escalator.reshape(-1)
        busy_agents_position = np.random.choice(a=self.busy_num, size=self.busy_num, replace=False, p=None) * 2
        idle_agents_position = (np.random.choice(a=self.init_area-self.busy_num, size=self.agent_num-self.busy_num,
                                                 replace=False, p=None) + self.busy_num) * 2
        l = idle_agents_position + 3
        self.escalator[np.concatenate((busy_agents_position, idle_agents_position), axis=-1)] = "occupied"
        self.escalator = self.escalator.reshape(shapeEsc)

        busy_count = 0
        idle_count = 0
        for i in range(self.agent_num):
            label = self.agents[i]["label"]
            if self.agents[i]['state'] == 'busy':
                self.agents[i]["position"] = [busy_agents_position[busy_count]//2, busy_agents_position[busy_count]%2]
                self.agents[i]["init_pos"] = busy_agents_position[busy_count]//2
                busy_count += 1
            else:
                self.agents[i]["position"] = [idle_agents_position[idle_count]//2, idle_agents_position[idle_count]%2]
                self.agents[i]["init_pos"] = idle_agents_position[idle_count]//2
                idle_count += 1

        # self.agents = sorted(self.agents, key=lambda a: a['position'][0])
        self.time_step = 0


    def init_agents(self):
        labels = np.random.choice(a=self.agent_num, size=self.agent_num, replace=False, p=None)
        busy_agents = np.random.choice(a=self.agent_num, size=self.busy_num, replace=False, p=None)
        self.reward = {}
        self.rewardRec = {}
        for i in range(len(self.agents)):
            a = self.agents[i]
            if i in busy_agents:
                a["state"] = "busy"
            else:
                a["state"] = "idle"

            a["label"] = i
            a["arrived"] = False
            a["reward"] = False
            self.reward[labels[i]] = 0
            self.rewardRec[labels[i]] = 0


    def check_superimposed(self):
        '''
        To see if any two of agents are on the same position,
        which is invalid
        '''
        positions = [a["position"] for a in self.agents]
        for i in range(len(positions)):
            for j in range(i+1, len(positions[i])):
                assert positions[i]!=positions[j], "agents superimposed: {num1, num2}".format(num1=i, num2=j)

    def check_valid(self, action, agent):
        '''
        to check if the action of an agent is valid, which is to check is the next
        the postion that the agent is going has already been occupied by others
        :param action: the current action of an given agent
        :return: True or False
        '''
        def next_pos(agent, action):
            cpos = agent["position"]   #get the current position of the given agent
            if action == "left":
                npos = [cpos[0], cpos[1] - 1]
                return npos
            if action == "right":
                npos = [cpos[0], cpos[1] + 1]
                return npos
            if action == "proceed" and agent['state'] == 'busy':
                npos = [cpos[0] + 1, cpos[1]]
                return npos
            if action == "stay":
                npos = cpos
                return npos
            return cpos

        next_position = next_pos(agent, ACTIONS[action])
        if agent['position'][0]<self.length-1 and next_position[1]>=0 and \
                next_position[1]<=1 and self.escalator[tuple(next_position)] != 'occupied':
            return True
        else:
            return False


    def change_position(self, action):
        '''
        execute the an agent's chosen action
        :param action: a tuple or list of agents' actions
        :return: None
        '''
        def next_pos(agent, action):
            cpos = agent["position"]   #get the current position of the given agent
            if action == "left":
                npos = [cpos[0], cpos[1] - 1]
                return npos
            if action == "right":
                npos = [cpos[0], cpos[1] + 1]
                return npos
            if action == "proceed":
                npos = [cpos[0] + 1, cpos[1]]
                return npos
            if action == "stay":
                npos = cpos
                return npos


        for i in range(len(self.agents)):
            if self.check_valid(action[i], self.agents[i]):
                self.escalator[tuple(self.agents[i]['position'])] = 'empty***'
                npos = next_pos(self.agents[i], ACTIONS[action[i]])
                self.agents[i]["position"] = npos
                self.escalator[tuple(npos)] = "occupied"

    def reward_cal(self):
        '''
        To calculate the reward each agent should receive
        :param a: an realized agent
        :return: the reward one (all) agent should receive
        '''
        for i in range(self.agent_num):
            a = self.agents[i]
            if a['state'] == 'busy' and a['arrived'] and self.rewardRec[i] == 0:
                self.reward[a['label']] = self.length - self.time_step - a['init_pos']
                self.rewardRec[a['label']] = self.reward[a['label']]
            elif a['arrived'] and self.reward[i] != 0:
                self.reward[a['label']] = 0

        return self.reward

    def _reward_cal2(self):
        for i in range(self.agent_num):
            a = self.agents[i]
            if a['state'] == 'busy':
                self.reward[a['label']] = -1
            if a['arrived'] and self.reward[i] != 0:
                self.reward[a['label']] = 0

        return self.reward

    def auto_proceed(self):
        '''
        since it's a escalator env, agents should be able to proceed
        forward automatically along with the move of the escalator.
        :return: None
        '''
        del self.escalator
        self.escalator = np.full((self.length, 2), 'empty***')
        for a in self.agents:
            if a['arrived'] == False:
                a["position"][0] = a["position"][0] + 1
                if a["position"][0] >= self.length:
                    a["arrived"] = True
                    print("{label}arrived,state:{state},time_step:{}".format(label=a['label'],state=a['state'],time_step=self.time_step))
                else:
                    self.escalator[tuple(a['position'])] = 'occupied'

    def auto_proceed2(self):
        '''
        not finished yet, the second version of the auto_proceed
        :return: None
        '''
        del self.escalator
        self.escalator = np.full((self.length, 2), 'empty***')
        for a in self.agents:
            if a['arrived'] == False:
                a['position'][0] = a['position'][0] + 1
                if a['position'][0] >= self.length - 1:
                    a['arrived'] = True
                    print("agent{label} arrived, state:{state}, time_step:{time_step}".format(label=a['label'],state=a['state'],time_step=self.time_step))
                else:
                    self.escalator[tuple(a['position'])] = 'occupied'

        for a in self.agents:
            if a['arrived'] == False and a['state'] == 'busy':
                if self.escalator[(a['position'][0]+1, a['position'][1])] == 'empty***':
                    last_pos = copy.deepcopy(a['position'])
                    a['position'][0] = a['position'][0] + 1
                    if a['position'][0] >= self.length:
                        a['arrived'] = True
                        print("{label}arrived,state:{state}".format(label=a['label'],state=a['state']))
                    else:
                        self.escalator[tuple(a['position'])] = 'occupied'
                        self.escalator[tuple(last_pos)] = 'empty***'


    @property
    def observation(self):
        func = lambda x:1 if x=='occupied' else 0
        temp = self.escalator
        temp = np.array([item for item in map(func, temp.reshape(-1))])
        return temp.reshape(-1, 2)

    def get_agent_pos(self, index):
        escalator = np.full(self.escalator.shape, 0)
        if not (self.agents[index]['position'][0] > 20):
            escalator[self.agents[index]['position']] = 1
        return escalator

    @property
    def done_info(self):
        return [agent['arrived'] for agent in self.agents]

    def agent_state(self, index):
        state = self.agents[index]['state']
        if state == 'busy':
            return np.array((1,))
        else:
            return np.array((0,))

    def _close_view(self):
        if self.root:
            self.root.destory()
            self.root = None
            self.canvas = None
        # self.done = True

    def _render(self):
        map = self.escalator
        scale = 30
        width = map.shape[0] * scale
        height = map.shape[1] * scale
        if self.root is None:
            self.root = tkinter.Tk()
            self.root.title("escalator env")
            self.root.protocol("WM_DELETE_WINDOW", self._close_view)
            self.canvas = tkinter.Canvas(self.root, width=width, height=height)
            self.canvas.pack()

        self.canvas.delete(tkinter.ALL)
        self.canvas.create_rectangle(0, 0, width, height, fill="black")

        def fill_cell(x, y, color):
            self.canvas.create_rectangle(
                x * scale,
                y * scale,
                (x + 1) * scale,
                (y + 1) * scale,
                fill=color
            )

        for x in range(map.shape[0]):
            for y in range(map.shape[1]):
                if map[x,y] == 'empty***':
                    fill_cell(x,y,'Grey')
                if map[x,y] == 'occupied':
                    fill_cell(x,y,'Green')

        for a in self.agents:
            if a['state'] == 'busy':
                fill_cell(a['position'][0],a['position'][1],'Red')

        fill_cell(self.length-1,0,'Purple')
        fill_cell(self.length-1,1,'Purple')

        self.root.update()

    def info(self, epoch=None):
        print("==================================================================================")
        print(" epoch:",epoch)
        print(" numbur of agents:%d\n"%self.agent_num,
              "length of escalator:%d\n"%(self.length-1),
              "number of busy agents:%d"%self.busy_num)
        print("----------------------------------------------------------------------------------")

    def get_idle_agents(self):
        labels = []
        for agent in self.agents:
            if agent['state'] == 'idle':
                labels.append(agent['label'])

        return labels


