'''
Author: Qi7
Date: 2022-07-19 00:26:02
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-10-13 22:25:57
Description: 
'''
#%%
from cProfile import label
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
from utils import heatmap, heatmap3D, UncertaintySampling, dataSampling, removeOutBound, plot_true_bound

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
        fig_path = "savedModel/random_sample/figs/"
    elif method == 1:
        model_path = "savedModel/active_learning/"
        fig_path = "savedModel/active_learning/figs/"
    elif method == 2:
        model_path = "savedModel/theoretical/"
        fig_path = "savedModel/theoretical/figs/"
    else:
        print("invalid input. Please check.")
        exit()

    # If transfer learning is enabled, load the pretrained model as the backbone.
    if pretrained:
        trained_model = "savedModel/active_learning/2d_epochs200_lr_0.001_bs_16_bestmodel_transfer.pth"
        
    if os.path.isdir(model_path) == False:
        os.makedirs(model_path)
        
    if os.path.isdir(fig_path) == False:
        os.makedirs(fig_path)
    
    # define the uncertainty methods
    sampler = UncertaintySampling()
    sample_method = sampler.least_confidence
    
    # Specify the dataset
    X_csv = "dataset/flexibility/2d/2d_ratio_1.csv"
    y_csv = "dataset/flexibility/2d/2d_isfeas_1.csv"
    sample_upper_bound = "dataset/flexibility/2d/2d_upper_1.csv"
    sample_lower_bound = "dataset/flexibility/2d/2d_lower_1.csv"
    theory_upper_bound = "dataset/flexibility/2d/2d_theory_upper_1.csv"
    theory_lower_bound = "dataset/flexibility/2d/2d_theory_lower_1.csv"
    
    df_x = pd.read_csv(X_csv, header=None)
    df_y = pd.read_csv(y_csv, header=None)
    upper = pd.read_csv(sample_upper_bound, header=None)
    lower = pd.read_csv(sample_lower_bound, header=None)
    theory_upper = pd.read_csv(theory_upper_bound, header=None)
    theory_lower = pd.read_csv(theory_lower_bound, header=None)
    
    X1_u = upper[0][0]
    X2_u = upper[0][1]
    
    x1_tu = theory_upper[0][0]
    x2_tu = theory_upper[0][1]
    
    X1_l = lower[0][0]
    X2_l = lower[0][1]
    
    x1_tl = theory_lower[0][0]
    x2_tl = theory_lower[0][1]
    
    X = df_x.to_numpy()
    y = df_y.to_numpy()
    
    # The formula to retrieve the actual value from the ratio:
    # value = r * u + (1 - r) * ell
    X[:,0] = X[:,0] * X1_u + (1 - X[:,0]) * X1_l
    X[:,1] = X[:,1] * X2_u + (1 - X[:,1]) * X2_l
    
    
    # Plot the figure of raw data distribution
    if verbose:
        plt.figure(figsize=(19, 16))
        plt.title('Raw data distribution')
        plt.scatter(X[:,0], X[:,1], c=y)
        
        # plot theoretical bound
        plt.plot()
        plt.plot([x1_tl, x1_tu], [x2_tl, x2_tl], c='r', linewidth=5)
        plt.plot([x1_tl, x1_tu], [x2_tu, x2_tu], c='r', linewidth=5)
        plt.plot([x1_tl, x1_tl], [x2_tl, x2_tu], c='r', linewidth=5)
        plt.plot([x1_tu, x1_tu], [x2_tl, x2_tu], c='r', linewidth=5)
        plt.savefig(fig_path + "raw_distribution")
        plt.show()

    #Preprocessing
    
    # The number of samples for the initial training (epoch 1)
    num_init_samples = 100

    # The number of samples per epoch or iteration
    num_samples_per_epoch = 10

    # 80% data for training, 20% for testing
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
    upper_bound = [x1_tu, x2_tu]
    lower_bound = [x1_tl, x2_tl]
    upper_bound = (upper_bound - train_mean) / train_std
    lower_bound = (lower_bound - train_mean) / train_std
    

    
    # theoretical_bound
    tb = None
    if method == 2:
        tb = {
            "x1_hi": upper_bound[0],
            "x1_lo": lower_bound[0],
            "x2_hi": upper_bound[1],
            "x2_lo": lower_bound[1]
        }

    # Plot the figure of normalized data distribution
    plt.figure(figsize=(19, 16))
    plt.title('Raw data distribution (normalized)', fontsize=30)
    plt.scatter(X_all[:,0], X_all[:,1],c=y)
    plt.xlabel("feature 1", fontsize=18)
    plt.ylabel("feature 2", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # plot the theoretical boundary
    plt.plot([lower_bound[0], upper_bound[0]], [lower_bound[1], lower_bound[1]], c='r', linewidth=5)
    plt.plot([lower_bound[0], upper_bound[0]], [upper_bound[1], upper_bound[1]], c='r', linewidth=5)
    plt.plot([lower_bound[0], lower_bound[0]], [lower_bound[1], upper_bound[1]], c='r', linewidth=5)
    plt.plot([upper_bound[0], upper_bound[0]], [lower_bound[1], upper_bound[1]], c='r', linewidth=5)
    
    plot_true_bound()
    
    plt.savefig(fig_path + "raw_distribution_normalized")
    if verbose:
        plt.show()
    else:
        plt.close()


    # Current data pool for sampling
    train_data_pool = np.append(X_train, y_train, 1)
    test_data_pool = np.append(X_test, y_test, 1)

   
    # in this case we only test the data in the theoritical bound
    if method == 2:
        test_data_pool = removeOutBound(tb=tb, data=test_data_pool)


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
    
    # Parameter for SGD optimizer
    optimizer_momentum = 0.9
    if pretrained:
        num_frozen_layers = 3 # the number of layers to be frozen

    # parameter for the learning rate scheduler
    lr_lambda = lambda epoch: 1 ** epoch 
    # Specify the scheduler, using LambdaLr in this case. More info: 
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
        # for the first epoch
        if t == 0:
            if verbose: print(f"The initial round, {num_init_samples} numbers of sample have been randomly sampled")
            np.random.shuffle(train_data_pool)
            cur_training_data = train_data_pool[0:num_init_samples, :]
            sampled_data_pool = np.append(sampled_data_pool, cur_training_data, axis=0)
            train_data_pool = np.delete(train_data_pool, obj=slice(0, num_init_samples), axis=0)
            if verbose:
                    plt.figure(figsize=(19, 16))
                    plt.title('Data distribution with physical theoritic data boundary')
                    plt.scatter(X_all[:,0], X_all[:,1], c=y)
                    plt.scatter(sampled_data_pool[:,0], sampled_data_pool[:,1], c='r', marker='v', label="sample data pool")
                    # plot the theoretical boundary
                    plt.plot([lower_bound[0], upper_bound[0]], [lower_bound[1], lower_bound[1]], c='r', linewidth=5)
                    plt.plot([lower_bound[0], upper_bound[0]], [upper_bound[1], upper_bound[1]], c='r', linewidth=5)
                    plt.plot([lower_bound[0], lower_bound[0]], [lower_bound[1], upper_bound[1]], c='r', linewidth=5)
                    plt.plot([upper_bound[0], upper_bound[0]], [lower_bound[1], upper_bound[1]], c='r', linewidth=5)
                    plt.legend()
                    plt.show()
            if method == 2:
                # After inital randomly sample, the train data pool will only include data in the uncertaintly area
                train_data_pool = removeOutBound(tb=tb, data=train_data_pool)
                if verbose:
                    plt.figure(figsize=(19, 16))
                    plt.title('Data distribution with physical theoritic data boundary')
                    plt.scatter(X_all[:,0], X_all[:,1], c=y)
                    plt.scatter(train_data_pool[:,0], train_data_pool[:,1], c='r', marker='v', label="training data pool")
                    # plot the theoretical boundary
                    plt.plot([lower_bound[0], upper_bound[0]], [lower_bound[1], lower_bound[1]], c='r', linewidth=5)
                    plt.plot([lower_bound[0], upper_bound[0]], [upper_bound[1], upper_bound[1]], c='r', linewidth=5)
                    plt.plot([lower_bound[0], lower_bound[0]], [lower_bound[1], upper_bound[1]], c='r', linewidth=5)
                    plt.plot([upper_bound[0], upper_bound[0]], [lower_bound[1], upper_bound[1]], c='r', linewidth=5)
                    plt.legend()
                    plt.show()
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

        # in this case, print heatmap every 100 iterations
        if (t+1) % 5 == 0:
            heatmap(model=model, dataset=X_all, sampled_data=cur_training_data, device=device, uncertainty_methods=sample_method, epoch=t+1, method=method, path=fig_path, lb=lower_bound, ub=upper_bound ,tb=tb, pretrained=pretrained, verbose=verbose)
        if pretrained:
            suffix = "_transfer"
        else:
            suffix = ""
        if max_loss > history['test_loss'][-1]:
            max_loss = history['test_loss'][-1]
            torch.save(model.state_dict(), model_path + f"2d_epochs{epochs}_lr_{Lr}_bs_{bs}_bestmodel{suffix}.pth")

    time_delta = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_delta // 60, time_delta % 60))

    # save history file
    np.save(model_path + f"2d_epochs{epochs}_lr_{Lr}_bs_{bs}_history{suffix}.npy", history)

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