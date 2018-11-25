#!/usr/bin/env python

import os, numpy as np
from time import time
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import pandas as pd

from PIL import Image
from collections import Counter



class DataLoader(data.Dataset):
    def __init__(self, transitions):
        self.transitions = transitions
        self.N = len(self.transitions)

    def __getitem__(self, index):
        trx = self.transitions[index]
        state = trx[0]
        action = trx[1]
        reward = trx[2]
        next_state = trx[3]

        #state
        state = Variable(torch.Tensor(state))

        #action
        action = self.to_onehot(4, action)
        action = Variable(torch.Tensor(action))

        #reward
        reward = self.reward_onehot(reward)
        reward = Variable(torch.Tensor(reward))

        #next state
        next_state = Variable(torch.Tensor(next_state))

        return state, action, reward, next_state

    def reward_onehot(self, reward):
        my_onehot = np.zeros((3))
        r = {-1:0, -50:1, 50:2}
        my_onehot[r[reward]] = 1.0
        return my_onehot

    def to_onehot(self, size, value):
        my_onehot = np.zeros((size))
        my_onehot[value] = 1.0
        return my_onehot

    def __len__(self):
        return self.N
