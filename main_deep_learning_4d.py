'''
Author: Qi7
Date: 2022-07-19 00:26:02
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-10-16 16:40:46
Description: 
'''
#%%
from email import header
import os
import numpy as np
import time
import pandas as pd
import torch
import warnings
from torch.utils.data import DataLoader
from dataloaders import MyLoader
from torchinfo import summary
from models import FNN, FNN_4d
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score
from matplotlib import pyplot as plt
from training import train_loop, eval_loop
from torch.optim import lr_scheduler
from utils import heatmap, heatmap3D, UncertaintySampling, dataSampling, removeOutBound_4d

# Ignore the warning information
warnings.filterwarnings('always')


def main(verbose=False, method=0, pretrained=False):
    """
    verbose: 
        If True, print detailed debug information.
        
    method:
        Choose sample strategies 
        
        0-randomly sampling, 
        1-active learning, 
        2-active learning with physical information
    
    pretrained:
        If pretrained = True, transfer learning is being used.
    
    Formula for converting ratio to sample value:
    value = r * u + (1 - r) * ell, 
    r: ratio
    sample upper bound: u
    sample lower bound: ell
    
    """
    if method == 0:
        model_path = "savedModel/random_sample/"
    elif method == 1:
        model_path = "savedModel/active_learning/"
    elif method == 2:
        model_path = "savedModel/theoretical/"
    else:
        print("invalid input. Please check.")
        exit()

    # if transfer learning is enabled, load the pretrained model as the backbone
    if pretrained:
        trained_model = "savedModel/theoretical/4d_1_epochs200_lr_0.001_bs_16_bestmodel.pth"
        
    if os.path.isdir(model_path) == False:
        os.makedirs(model_path)
    
    # define the uncertainty methods
    sampler = UncertaintySampling()
    sample_method = sampler.least_confidence
    
    # The following specify the dataset
    timestamp = '7'
    
    X_csv = f"dataset/flexibility/4d/4d_ratio_{timestamp}.csv"
    y_csv = f"dataset/flexibility/4d/4d_isfeas_{timestamp}.csv"
    sample_upper_bound = f"dataset/flexibility/4d/4d_upper_{timestamp}.csv"
    sample_lower_bound = f"dataset/flexibility/4d/4d_lower_{timestamp}.csv"
    theory_upper_bound = f"dataset/flexibility/4d/4d_theory_upper_{timestamp}.csv"
    theory_lower_bound = f"dataset/flexibility/4d/4d_theory_lower_{timestamp}.csv"
    
    df_x = pd.read_csv(X_csv, header=None)
    # X = df.to_numpy()
    df_y = pd.read_csv(y_csv, header=None)
    # y = df.to_numpy()
    upper = pd.read_csv(sample_upper_bound, header=None)
    lower = pd.read_csv(sample_lower_bound, header=None)
    theory_upper = pd.read_csv(theory_upper_bound, header=None)
    theory_lower = pd.read_csv(theory_lower_bound, header=None)
    
    # solve data unbalance problem for 4-dimensional case:
    
    coef = 1.2 # number of unfeasible sampels = coef * number of feasible samples
    
    cat_df = pd.concat([df_x, df_y], axis=1)
    cat_df.columns = ['x1','x2','x3','x4','y']
    # the number of samples being removed for solving the unbalance issues
    num_of_remove = cat_df["y"].value_counts()[0] - int(coef * cat_df["y"].value_counts()[1])
    
    # shuffle the dataframe
    cat_df = cat_df.sample(frac=1, random_state=1)
    cat_df.drop(cat_df[cat_df["y"] == 0].index[:num_of_remove], inplace=True)
    
    X = cat_df[['x1','x2','x3','x4']].to_numpy()
    y = cat_df["y"].to_numpy().reshape(-1,1)
    
    X1_u = upper[0][0]
    X2_u = upper[0][1]
    X3_u = upper[0][2]
    X4_u = upper[0][3]
    
    x1_tu = theory_upper[0][0]
    x2_tu = theory_upper[0][1]
    x3_tu = theory_upper[0][2]
    x4_tu = theory_upper[0][3]
    
    X1_l = lower[0][0]
    X2_l = lower[0][1]
    X3_l = lower[0][2]
    X4_l = lower[0][3]
    
    x1_tl = theory_lower[0][0]
    x2_tl = theory_lower[0][1]
    x3_tl = theory_lower[0][2]
    x4_tl = theory_lower[0][3]
    
    # The formula to retrieve the actual value:
    # value = r * u + (1 - r) * ell
    X[:,0] = X[:,0] * X1_u + (1 - X[:,0]) * X1_l
    X[:,1] = X[:,1] * X2_u + (1 - X[:,1]) * X2_l
    X[:,2] = X[:,2] * X3_u + (1 - X[:,2]) * X3_l
    X[:,3] = X[:,3] * X4_u + (1 - X[:,3]) * X4_l
    

    #%% preprocessing
    # The number of samples for the initial training.
    num_init_samples = 100

    # The number of samples per epoch or iteration
    num_samples_per_epoch = 10

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

    # Number of input features
    num_features = X_train.shape[1]

    # The data pool with data has been sampled (features + label)
    sampled_data_pool = np.empty((0, X_train.shape[1]+1))

    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)

    # Feature data normalization
    X_all = (X - train_mean) / train_std
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    
    # Theoritical bound normalization
    upper_bound = [x1_tu, x2_tu, x3_tu, x4_tu]
    lower_bound = [x1_tl, x2_tl, x3_tl, x4_tl]
    upper_bound = (upper_bound - train_mean) / train_std
    lower_bound = (lower_bound - train_mean) / train_std
    
    # theoretical_bound
    tb = None
    if method == 2:
        tb = {
            "x1_hi": upper_bound[0],
            "x1_lo": lower_bound[0],
            "x2_hi": upper_bound[1],
            "x2_lo": lower_bound[1],
            "x3_hi": upper_bound[2],
            "x3_lo": lower_bound[2],
            "x4_hi": upper_bound[3],
            "x4_lo": lower_bound[3],
        }

    # Current data pool for sampling
    train_data_pool = np.append(X_train, y_train, 1)
    test_data_pool = np.append(X_test, y_test, 1)

   
    # in this case we only test the data in the theoritical bound
    if method == 2:
        test_data_pool = removeOutBound_4d(tb=tb, data=test_data_pool)


    cur_X_test, cur_y_test = test_data_pool[:, 0:num_features], test_data_pool[:, -1].reshape(-1, 1)
    current_data_test = MyLoader(data_root=cur_X_test, data_label=cur_y_test)
    test_dataloader = DataLoader(current_data_test, batch_size = 16, shuffle = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FNN(n_inputs=num_features)

    # load the transfered model
    if pretrained:
        model.load_state_dict(torch.load(trained_model))
    model.to(device)

    # hyper parameters
    epochs = 200
    Lr = 0.001
    optimizer_momentum = 0.9
    if pretrained:
        num_frozen_layers = 3 # the number of layers to be frozen

    # parameter for the learning rate scheduler
    lr_lambda = lambda epoch: 1 ** epoch 
    # sepcify the scheduler, using LambdaLr in this case. More info: 
    # https://pytorch.org/docs/stable/optim.html
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR 

    loss_fn = torch.nn.BCELoss()
    metric_fn = accuracy_score
    bs = 16
    
    # configure the optimizer for training.
    if pretrained == False:
        optimizer = torch.optim.SGD(model.parameters(), lr=Lr, momentum=optimizer_momentum)
        # optimizer = torch.optim.Adam(model.parameters(), lr=Lr)
    # if transfer learning is used
    else:
        # frozen the layer no need to train
        cnt = 0
        for weights in model.parameters():
            if cnt >= 2 * num_frozen_layers: break
            weights.requires_grad = False
            cnt += 1
        
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.SGD(params, lr=Lr, momentum=optimizer_momentum)
        # optimizer = torch.optim.Adam(params, lr=Lr)

    scheduler = lr_scheduler(optimizer, lr_lambda=lr_lambda)
    

    summary(model, input_size=(64, 1, num_features), verbose=1)
    history = dict(train_loss=[], test_loss=[], acc_train=[], acc_test=[], f1_train=[], f1_test=[])
    max_loss = 1
    start = time.time()
    
    for t in range(epochs):
        print(f"Epoch {t + 1}/{epochs}, learning rate {scheduler.get_lr()}")\
        # for first epoch
        if t == 0:
            if verbose: print(f"The initial round, {num_init_samples} numbers of sample have been randomly sampled")
            np.random.shuffle(train_data_pool)
            cur_training_data = train_data_pool[0:num_init_samples, :]
            sampled_data_pool = np.append(sampled_data_pool, cur_training_data, axis=0)
            train_data_pool = np.delete(train_data_pool, obj=slice(0, num_init_samples), axis=0)
            
            if method == 2:
                # After inital randomly sample, the train data pool will only include data in the uncertaintly area
                train_data_pool = removeOutBound_4d(tb=tb, data=train_data_pool)
        else:
            if verbose: print(f"The sampling round, {num_samples_per_epoch} numbers of sample have been randomly sampled")

            # if randomly sample
            if method == 0:
                np.random.shuffle(train_data_pool)
                cur_training_data = train_data_pool[0:num_samples_per_epoch, :]
                sampled_data_pool = np.append(sampled_data_pool, cur_training_data, axis=0)
                train_data_pool = np.delete(train_data_pool, obj=slice(0, num_samples_per_epoch), axis=0)
            # if active learning sample
            else:
                train_data_pool = dataSampling(model=model, uncertainty_methods=sample_method, data_pool=train_data_pool, device=device)
                cur_training_data = train_data_pool[0:num_samples_per_epoch, :]
                sampled_data_pool = np.append(sampled_data_pool, cur_training_data, axis=0)
                train_data_pool = np.delete(train_data_pool, obj=slice(0, num_samples_per_epoch), axis=0)

        cur_X_train, cur_y_train = sampled_data_pool[:, 0:num_features], sampled_data_pool[:, -1].reshape(-1, 1)
        current_data = MyLoader(data_root=cur_X_train, data_label=cur_y_train)
        train_dataloader = DataLoader(current_data, batch_size = bs, shuffle = True)

        print('-' * 40)
        train_loop(
            trainLoader=train_dataloader, 
            model=model, 
            device=device, 
            optimizer=optimizer, 
            lr_scheduler=scheduler,
            metric_fn=metric_fn,
            loss_fn=loss_fn,
            history=history,
            verbose=verbose
        )
        # update the learning rate per epoch, if wanna update lr per batch. add this line in train_loop function
        scheduler.step()

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

        if pretrained:
            suffix = "_transfer"
        else:
            suffix = ""
        if max_loss > history['test_loss'][-1]:
            max_loss = history['test_loss'][-1]
            torch.save(model.state_dict(), model_path + f"4d_{timestamp}_epochs{epochs}_lr_{Lr}_bs_{bs}_bestmodel{suffix}.pth")

    time_delta = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_delta // 60, time_delta % 60))

    # save history file
    np.save(model_path + f"4d_{timestamp}_epochs{epochs}_lr_{Lr}_bs_{bs}_history{suffix}.npy", history)

    if verbose:
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
    main(verbose=False, method=2, pretrained=True)