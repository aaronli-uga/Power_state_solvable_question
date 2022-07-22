'''
Author: Qi7
Date: 2022-07-19 00:26:02
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-07-22 02:24:06
Description: 
'''
#%%
import os
import numpy as np
import copy
import time
import pandas as pd
import torch
import copy
from torch.utils.data import DataLoader
from dataloaders import MyLoader
from torchinfo import summary
from models import FNN, new_FNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score
from matplotlib import pyplot as plt
from training import train_loop, eval_loop
from torch.optim import lr_scheduler


X_csv = "mod_ratio_10k.csv"
y_csv = "iffeas_10k.csv"

df = pd.read_csv(X_csv)
X = df.to_numpy()
df = pd.read_csv(y_csv)
y = df.to_numpy()

#%% preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std


training_data = MyLoader(data_root=X_train, data_label=y_train)
testing_data = MyLoader(data_root=X_test, data_label=y_test)

train_dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)
test_dataloader = DataLoader(testing_data, batch_size = 1024, shuffle = False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FNN(n_inputs=X_train.shape[1])
model.to(device)

epochs = 100
Lr = 0.01
loss_fn = torch.nn.BCELoss()
metric_fn = f1_score

summary(model, input_size=(64, 1, X_train.shape[1]), verbose=1)
history = dict(train=[], val=[], f1_train=[], f1_test=[])

start = time.time()
for t in range(epochs):
    print(f"Epoch {t + 1}/{epochs}, learning rate {Lr}")
    print('-' * 20)
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

time_delta = time.time() - start
print('Training complete in {:.0f}m {:.0f}s'.format(time_delta // 60, time_delta % 60))

#%%
plt.figure(figsize=(10,8))
plt.plot(history['val'])
plt.show()
plt.figure(figsize=(10,8))
plt.plot(history['f1_train'])
plt.show()
# %%
