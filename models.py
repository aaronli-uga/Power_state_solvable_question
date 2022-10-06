'''
Author: Qi7
Date: 2022-07-18 00:02:08
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-10-04 11:17:48
Description: 
'''
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import isInBound

# classical Fully connected neural network (FNN) model.
# same structure of the reference active learning paper
class FNN(nn.Module):

    def __init__(self, n_inputs):
        """
        tb: theoritical bound
        """
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 50)
        self.fc5 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dout = nn.Dropout(0.2)
        # self.tb = tb
    
    def forward(self, input):
        x = self.relu(self.fc1(input))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.dout(x)
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x


class FNN_4d(nn.Module):

    def __init__(self, n_inputs):
        """
        tb: theoritical bound
        """
        super(FNN_4d, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 64)
        self.fc9 = nn.Linear(64, 32)
        self.fc10 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        x = self.relu(self.fc1(input))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x = self.relu(self.fc9(x))
        x = self.sigmoid(self.fc10(x))
        return x