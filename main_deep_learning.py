'''
Author: Qi7
Date: 2022-07-19 00:26:02
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-07-19 01:18:01
Description: 
'''
import os
import numpy as np
import copy
import pandas as pd
from torch.utils.data import DataLoader
from torchinfo import summary
from models import FNN
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import torch

X_csv = "mod_ratio_10k.csv"
y_csv = "iffeas_10k.csv"

df = pd.read_csv(X_csv)
X = df.to_numpy()
df = pd.read_csv(y_csv)
y = df.to_numpy()
y = y.flatten()

#%% preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FNN(n_inputs=X_train.shape[1])
model.to(device)

epochs = 10
Lr = 0.01 
loss_fn = torch.nn.BCELoss()