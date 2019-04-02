from collections import deque
import random
import atari_py
import cv2
import torch
import numpy as np


from sc2env.environments.zone_intruders import ZoneIntrudersEnvironment

class Env():
    def __init__(self, args):
        self.env = ZoneIntrudersEnvironment(render=False)
        self.episode_length = 0
        self.total_episodes = 0

    def action_space(self):
        return self.env.action_space.n

    def reset(self):
        state = self.env.reset()
        self.episode_length = 0
        if self.total_episodes > 10:
            self.env = ZoneIntrudersEnvironment(render=False)
            self.total_episodes = 0
        # Context: three states
        self.prev_state_1 = state[1]
        self.prev_state_2 = state[1]
        state_with_context = np.concatenate([
            self.prev_state_1, self.prev_state_2, state[1]])
        return torch.Tensor(state_with_context).cuda()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.episode_length += 1
        if self.episode_length > 200:
            done = True
        if done:
            self.total_episodes += 1

        state_with_context = np.concatenate([
            self.prev_state_1, self.prev_state_2, state[1]])
        self.prev_state_1 = self.prev_state_2
        self.prev_state_2 = state[1]
        return torch.Tensor(state_with_context).cuda(), reward, done

    def train(self):
        pass

    def eval(self):
        pass

    def close(self):
        pass
