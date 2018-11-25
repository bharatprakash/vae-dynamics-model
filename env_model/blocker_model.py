#!/usr/bin/env python
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import sys
import os
import numpy as np

sys.path.append('..')
from data.blocker_dataloader import DataLoader

def to_onehot(size, value):
    my_onehot = np.zeros((size))
    my_onehot[value] = 1.0
    return my_onehot

class BlockerNet(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_size2):
        super(BlockerNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 32, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

        self.fc3 = nn.Linear(hidden_size2, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        h = self.relu(self.fc2(out))
        return h

    def forward(self, x, a):
        h = self.encode(x)
        out = torch.cat([h,a], 1)
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out


class BlockerModel(object):
    def __init__(self):
        self.model = BlockerNet(128,20,24).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss = nn.BCELoss().cuda()
        self.batch_size = 100

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def init_dataloader(self, transitions):
        self.train_data = DataLoader(transitions)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data,
                                            batch_size=self.batch_size,
                                            shuffle=True)

    def train(self, epochs):
        self.model.train()
        for epoch in range(1, epochs + 1):
            train_loss = 0
            for i, (states, actions, safe) in enumerate(self.train_loader):

                states = Variable(states).cuda()
                actions = Variable(actions).type(torch.FloatTensor).cuda()
                safe = Variable(safe).type(torch.FloatTensor).cuda()

                self.optimizer.zero_grad()
                safe_hat = self.model(states, actions)
                loss = self.loss(safe_hat, safe)

                loss.backward()
                train_loss += loss.data[0]
                self.optimizer.step()

            print('====> Epoch: {} Average loss: {:.4f}'.format(
                  epoch, train_loss / self.train_data.N))
            if epoch % 50 == 0:
                torch.save(self.model.state_dict(), './saved_models/blocker_model_pg.pth')

    def predict(self, state, action):
        self.model.eval()
        state = Variable(torch.Tensor(state.reshape((1,3,32,32)))).cuda()
        action = to_onehot(4, action)
        action = Variable(torch.Tensor(action.reshape((1, 4)))).cuda()
        safe = self.model(state, action)
        return safe
