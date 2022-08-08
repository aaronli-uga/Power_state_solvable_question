'''
Author: Qi7
Date: 2022-07-18 00:02:08
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-08-08 15:35:39
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

    def __init__(self, n_inputs, tb):
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
        # if_in_bound = isInBound(tb=self.tb, data=input)
        x = self.relu(self.fc1(input))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.dout(x)
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x