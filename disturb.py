'''
Author: Qi7
Date: 2023-05-16 00:02:31
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-20 22:49:53
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

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
        
data_path = "dataset/noise/"
files = [("X_3p_0519.csv", "isfeas_3p_0519.csv"),
         ("X_10p_0519.csv", "isfeas_10p_0519.csv"), 
         ("X_20p_0519.csv", "isfeas_20p_0519.csv"),
         ("X_30p_0519.csv", "isfeas_30p_0519.csv"),
         ("X_40p_0519.csv", "isfeas_40p_0519.csv")]

# X_csv = "dataset/noise/X_01p_new.csv"
# y_csv = "dataset/noise/isfeas_01p_new.csv"
f1_all, accuracy_all = [], []

for (X_csv, y_csv) in files:
    X_csv = data_path + X_csv
    y_csv = data_path + y_csv
    df_x = pd.read_csv(X_csv, header=None)
    df_y = pd.read_csv(y_csv, header=None)
    # for normalization
    train_mean = np.load("new_train_mean.npy")
    train_std = np.load("new_train_std.npy")

    X = df_x.to_numpy()
    y = df_y.to_numpy()

    print(train_mean, train_std)
    print(X.mean(axis=0), X.std(axis=0))
    X_test = (X - train_mean) / train_std
    # X_test = (X - X.mean(axis=0)) / X.std(axis=0)
    y_test = y

    history = dict(train_loss=[], test_loss=[], acc_train=[], acc_test=[], f1_train=[], f1_test=[])
    trained_model = "savedModel/active_learning/new_noise_2d_epochs120_lr_0.001_bs_16_bestmodel.pth"
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
        
    print(history['f1_test'][0])
    print(history['acc_test'][0])
    f1_all.append(float(format(history['f1_test'][0], '.3f')))
    accuracy_all.append(history['acc_test'][0])


print(f"F1 score: {f1_all}")
print(f"Accuracy: {accuracy_all}")

# objects = ('3%', '10%', '20%', '30%', '40%')
# y_pos = np.arange(len(objects))

# plt.bar(y_pos, accuracy_all, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Test Accuracy')
# plt.title('Noise level in percentage')

# plt.show()

objects = ('3%', '10%', '20%', '30%', '40%')
y_pos = np.arange(len(objects))

plt.bar(y_pos, f1_all, align='center', alpha=0.5)
addlabels(y_pos, f1_all)
plt.xticks(y_pos, objects)
plt.xlabel('Noise levels in percentage')
plt.ylabel('Test F1 score')
plt.title('F1 score in different levels of noise')

plt.show()