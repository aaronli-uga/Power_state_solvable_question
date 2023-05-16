'''
Author: Qi7
Date: 2023-05-16 00:02:31
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-16 00:35:24
Description: test the disturbance in DER.
'''
import torch 
import numpy as np
from torch.utils.data import DataLoader
from dataloaders import MyLoader
from models import FNN
from training import eval_loop
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd

X_csv = "dataset/noise/X_20p.csv"
y_csv = "dataset/noise/isfeas_20p.csv"
df_x = pd.read_csv(X_csv, header=None)
df_y = pd.read_csv(y_csv, header=None)
# for normalization
train_mean = np.load("train_mean.npy")
train_std = np.load("train_std.npy")

X = df_x.to_numpy()
y = df_y.to_numpy()

# X_test = (X - train_mean) / train_std
X_test = (X - X.mean()) / X.std()
y_test = y

history = dict(train_loss=[], test_loss=[], acc_train=[], acc_test=[], f1_train=[], f1_test=[])
trained_model = "savedModel/active_learning/2d_epochs120_lr_0.001_bs_16_bestmodel.pth"
num_features = X_test.shape[1]

test_data_pool = np.append(X_test, y_test, 1)
cur_X_test, cur_y_test = test_data_pool[:, 0:num_features], test_data_pool[:, -1].reshape(-1, 1)
current_data_test = MyLoader(data_root=cur_X_test, data_label=cur_y_test)
test_dataloader = DataLoader(current_data_test, batch_size = 512, shuffle = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FNN(n_inputs=num_features)
model.load_state_dict(torch.load(trained_model))
model.to(device)

loss_fn = torch.nn.BCELoss()
metric_fn = accuracy_score


eval_loop(
    dataloader=test_dataloader,
    model=model,
    epoch=100,
    loss_fn=loss_fn,
    metric_fn=metric_fn,
    device=device,
    history=history,
    verbose=False
)
    
print(history['f1_test'])
print(history['acc_test'])