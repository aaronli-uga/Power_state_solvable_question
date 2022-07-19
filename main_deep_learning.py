'''
Author: Qi7
Date: 2022-07-19 00:26:02
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-07-19 09:23:25
Description: 
'''
import os
import numpy as np
import copy
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataloaders import MyLoader
from torchinfo import summary
from models import FNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from training import get_accuracy, train_loop, eval_loop


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


bs = 100 #batch size
training_data = MyLoader(data_root=X_train, data_label=y_train)
testing_data = MyLoader(data_root=X_test, data_label=y_test)

train_dataloader = DataLoader(training_data, batch_size = bs, shuffle = True)
test_dataloader = DataLoader(testing_data, batch_size = bs, shuffle = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FNN(n_inputs=X_train.shape[1])
model.to(device)

epochs = 200
Lr = 0.002 
loss_fn = torch.nn.BCELoss()
metric_fn = get_accuracy

summary(model, input_size=(bs, 1, X_train.shape[1]), verbose=1)
history = dict(train=[], val=[], f1_train=[], f1_test=[])

for t in range(epochs):
    print(f"Epoch {t + 1} learning rate {Lr} \n ----------------")
    train_loop(
        trainLoader=train_dataloader, 
        model=model, 
        device=device, 
        LR=Lr, 
        metric_fn=metric_fn,
        loss_fn=loss_fn,
        history=history
    )
    eval_loop(
        dataloader=test_dataloader,
        model=model,
        epoch=t,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        device=device,
        history=history
    )

plt.figure(figsize=(10,8))
plt.plot(history['val'])
plt.show()
plt.figure(figsize=(10,8))
plt.plot(history['f1_train'])
plt.show()