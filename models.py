'''
Author: Qi7
Date: 2022-07-18 00:02:08
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-07-19 01:17:05
Description: 
'''
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

# classical Fully connected neural network (FNN) model.
# same structure of the reference active learning paper
class FNN(nn.Module):

    def __init__(self, n_inputs):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 50)
        self.fc5 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        output = self.relu(self.fc1(input))
        output = self.relu(self.fc2(output))
        output = self.relu(self.fc3(output))
        output = self.relu(self.fc4(output))
        output = self.sigmoid(self.fc5(output))
        return output