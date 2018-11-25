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
from data.dataloader import DataLoader

def to_img(x):
    x = x.view(x.size(0), 3, 32, 32)
    return x

def to_onehot(size, value):
    my_onehot = np.zeros((size))
    my_onehot[value] = 1.0
    return my_onehot


class ObsNet(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_size2):
        super(ObsNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 32, intermediate_size)
        # Latent space
        self.fc21 = nn.Linear(intermediate_size, hidden_size)
        self.fc22 = nn.Linear(intermediate_size, hidden_size)

        # Decoder
        self.fc3 = nn.Linear(hidden_size2, intermediate_size)
        self.fc4 = nn.Linear(intermediate_size, 8192)
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return h1, self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
        out = out.view(out.size(0), 32, 16, 16)
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.sigmoid(self.conv5(out))
        return out

    def forward(self, x, a):
        h1, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        za = torch.cat([z,a], 1)
        return self.decode(za), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.view(-1, 32 * 32 * 3),
                                     x.view(-1, 32 * 32 * 3), size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


class ObsModel(object):
    def __init__(self):
        self.model = ObsNet(128,20,24).cuda()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=1e-3)
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
            for i, (states, actions, rewards, next_states) in enumerate(self.train_loader):

                states = Variable(states).cuda()
                next_states = Variable(next_states).cuda()
                actions = Variable(actions).cuda()
                rewards = Variable(rewards).type(torch.LongTensor).cuda()

                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(states, actions)
                loss = self.model.loss_function(recon_batch, next_states, mu, logvar)
                loss.backward()
                train_loss += loss.data[0]
                self.optimizer.step()

            if epoch % 50 == 0:
                pred_pic = to_img(recon_batch.cpu().data)
                pred_pic = pred_pic[:8]
                save_image(pred_pic, './dc_img/image_{}.png'.format(epoch))
                true_pic = to_img(next_states.cpu().data)
                true_pic = true_pic[:8]
                save_image(true_pic, './dc_img/image_{}_t.png'.format(epoch))
                in_pic = to_img(states.cpu().data)
                in_pic = in_pic[:8]
                save_image(in_pic, './dc_img/image_{}_i.png'.format(epoch))

            if epoch % 25 == 0:
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                      epoch, train_loss/self.train_data.N))
            if epoch % 50 == 0:
                torch.save(self.model.state_dict(), './saved_models/obs_model.pth')

    def predict(self, state, action):
        self.model.eval()
        state = Variable(torch.Tensor(state.reshape((1,3,32,32)))).cuda()
        action = to_onehot(4, action)
        action = Variable(torch.Tensor(action.reshape((1, 4)))).cuda()
        next_state, mu, logvar = self.model(state, action)
        return next_state


#
