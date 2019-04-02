from collections import deque
import random
import atari_py
import cv2
import torch


from sc2env.environments.zone_intruders import ZoneIntrudersEnvironment

class Env():
    def __init__(self, args):
        self.env = ZoneIntrudersEnvironment()
        self.episode_length = 0
        self.total_episodes = 0

    def action_space(self):
        return self.env.action_space.n

    def reset(self):
        state = self.env.reset()
        self.episode_length = 0
        if self.total_episodes > 10:
            self.env = ZoneIntrudersEnvironment()
            self.total_episodes = 0
        return torch.Tensor(state[1]).cuda()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.episode_length += 1
        if self.episode_length > 200:
            done = True
        if done:
            self.total_episodes += 1
        return torch.Tensor(state[1]).cuda(), reward, done

    def train(self):
        pass

    def eval(self):
        pass

    def close(self):
        pass
