#!/usr/bin/env python

import os, numpy as np
from time import time
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch

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
        safe = trx[2]

        #state
        state = Variable(torch.Tensor(state))

        #action
        action = self.to_onehot(4, action)
        action = Variable(torch.Tensor(action))

        #safe
        #safe = self.to_onehot(2, safe)
        #safe = Variable(torch.Tensor(safe)).

        return state, action, safe

    def to_onehot(self, size, value):
        my_onehot = np.zeros((size))
        my_onehot[value] = 1.0
        return my_onehot

    def __len__(self):
        return self.N
