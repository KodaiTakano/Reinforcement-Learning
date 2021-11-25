import gym
import numpy as np
import random


class Agent:
    def __init__(self):
        self.Q = np.zeros((5**4, 2))
        self.last_s = None
        self.last_a = None

    # 値を5段階に量子化
    def quantize5(self, x, a, b):
        return 0 if x < -a else 1 \
            if x < -b else 2 \
            if x <= b else 3 \
            if x <= a else 4
