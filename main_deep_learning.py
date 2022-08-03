'''
Author: Qi7
Date: 2022-07-19 00:26:02
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-08-03 17:20:55
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
import warnings
from torch.utils.data import DataLoader
from dataloaders import MyLoader
from torchinfo import summary
from models import FNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score
from matplotlib import pyplot as plt
from training import train_loop, eval_loop
from torch.optim import lr_scheduler
from utils import heatmap, heatmap3D, UncertaintySampling, dataSampling

# Ignore the warning information
warnings.filterwarnings('always')


def main(verbose=False, method=0):
    """
    method: 0-randomly sampling, 1-active learning, 2-active learning with physical information
    """
    if method == 2:
        if verbose: print("configure the physical information for active learning")
        pass

    sampler = UncertaintySampling()
    sample_method = sampler.least_confidence
    # IEEE 39 bus
    # X_csv = "dataset/IEEE_39_bus/mod_ratio_10k.csv"
    # y_csv = "dataset/IEEE_39_bus/iffeas_10k.csv"

    # IEEE 118 bus 2d 
    X_csv = "dataset/IEEE_118_bus/mod_ratio_10k_2d.csv"
    y_csv = "dataset/IEEE_118_bus/iffeas_10k_2d.csv"

    # IEEE 118 bus 2d version2
    # X_csv = "dataset/IEEE_118_bus/mod_ratio_10k_2d_v2.csv"
    # y_csv = "dataset/IEEE_118_bus/iffeas_10k_2d_v2.csv"

    # IEEE 118 bus 3d
    # X_csv = "dataset/IEEE_118_bus/mod_ratio_20k_3d.csv"
    # y_csv = "dataset/IEEE_118_bus/iffeas_20k_3d.csv"

    df = pd.read_csv(X_csv)
    X = df.to_numpy()
    df = pd.read_csv(y_csv)
    y = df.to_numpy()

    #%% preprocessing

    # The number of samples for the initial training.
    num_init_samples = 1000

    # The number of samples per epoch or iteration
    num_samples_per_epoch = 100

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

    # number of input features
    num_features = X_train.shape[1]

    # the data pool with data has been sampled
    sampled_data_pool = np.empty((0, X_train.shape[1]+1))

    all_data_mean = X.mean(axis=0)
    all_data_std = X.std(axis=0)

    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)

    X_all = (X - all_data_mean) / all_data_std
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    all_data = MyLoader(data_root=X_all, data_label=y)
    training_data = MyLoader(data_root=X_train, data_label=y_train)
    testing_data = MyLoader(data_root=X_test, data_label=y_test)

    all_dataloader = DataLoader(all_data, batch_size = 1, shuffle = False)
    train_dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)
    test_dataloader = DataLoader(testing_data, batch_size = 1024, shuffle = False)
    draw_test_dataloader = DataLoader(testing_data, batch_size = 1, shuffle = False)

    # Current data pool for sampling
    train_data_pool = np.append(X_train, y_train, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNN(n_inputs=num_features)
    model.to(device)

    epochs = 10
    Lr = 0.01
    loss_fn = torch.nn.BCELoss()
    metric_fn = accuracy_score

    summary(model, input_size=(64, 1, num_features), verbose=1)
    history = dict(train_loss=[], test_loss=[], acc_train=[],  acc_test=[], f1_train=[], f1_test=[])

    start = time.time()
    for t in range(epochs):
        print(f"Epoch {t + 1}/{epochs}, learning rate {Lr}")
        if t == 0:
            if verbose: print(f"The initial round, {num_init_samples} numbers of sample have been randomly sampled")
            np.random.shuffle(train_data_pool)
            cur_training_data = train_data_pool[0:num_init_samples, :]
            sampled_data_pool = np.append(sampled_data_pool, cur_training_data, axis=0)
            train_data_pool = np.delete(train_data_pool, obj=slice(0, num_init_samples), axis=0)
        else:
            if verbose: print(f"The sampling round, {num_samples_per_epoch} numbers of sample have been randomly sampled")

            # if randomly sample
            if method == 0:
                np.random.shuffle(train_data_pool)
                cur_training_data = train_data_pool[0:num_samples_per_epoch, :]
                sampled_data_pool = np.append(sampled_data_pool, cur_training_data, axis=0)
                train_data_pool = np.delete(train_data_pool, obj=slice(0, num_samples_per_epoch), axis=0)
            # if active learning sample
            elif method == 1:
                print("test")
                train_data_pool = dataSampling(model=model, uncertainty_methods=sample_method, data_pool=train_data_pool, device=device)
                cur_training_data = train_data_pool[0:num_samples_per_epoch, :]
                sampled_data_pool = np.append(sampled_data_pool, cur_training_data, axis=0)
                train_data_pool = np.delete(train_data_pool, obj=slice(0, num_samples_per_epoch), axis=0)
            # if active learning sample with physical information
            elif method == 2:
                pass
            else:
                if verbose: print("invalid input")
                exit()
        cur_X_train, cur_y_train = sampled_data_pool[:, 0:num_features], sampled_data_pool[:, -1].reshape(-1, 1)
        current_data = MyLoader(data_root=cur_X_train, data_label=cur_y_train)
        train_dataloader = DataLoader(current_data, batch_size = 16, shuffle = True)

        print('-' * 40)
        train_loop(
            trainLoader=train_dataloader, 
            model=model, 
            device=device, 
            LR=Lr, 
            metric_fn=metric_fn,
            loss_fn=loss_fn,
            history=history,
            verbose=verbose
        )
        eval_loop(
            dataloader=test_dataloader,
            model=model,
            epoch=t,
            loss_fn=loss_fn,
            metric_fn=metric_fn,
            device=device,
            history=history,
            verbose=verbose
        )

        heatmap(model=model, dataset=X_all, device=device, uncertainty_methods=sample_method, epoch=t+1)
    # heatmap3D(model=model, dataset=X_all, dataloader=all_dataloader, device=device)
    # heatmap3D(model=model, dataset=X_test, dataloader=draw_test_dataloader, device=device)



    time_delta = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_delta // 60, time_delta % 60))

    #%%
    plt.figure(figsize=(10,8))
    plt.title('Train Loss')
    plt.plot(history['train_loss'])
    plt.show()

    plt.figure(figsize=(10,8))
    plt.title('Test Loss')
    plt.plot(history['test_loss'])
    plt.show()

    plt.figure(figsize=(10,8))
    plt.title('Train Accuracy')
    plt.plot(history['acc_train'])
    plt.show()

    plt.figure(figsize=(10,8))
    plt.title('Test Accuracy')
    plt.plot(history['acc_test'])
    plt.show()

    plt.figure(figsize=(10,8))
    plt.title('Test F1')
    plt.plot(history['f1_test'])
    plt.show()
    # %%


if __name__ == '__main__':
    main(verbose=True, method=1)