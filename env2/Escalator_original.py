import Escalator
from BaseEnv import BaseEnv


class Base(BaseEnv):
    def __init__(self, args):
        super.__init__(args)

    def change_position(self, action):


class Escalator_original(Escalator.EscalatorEnv):
    def __init__(self, args):
        super.__init__(args)