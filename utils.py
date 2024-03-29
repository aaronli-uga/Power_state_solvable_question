'''
Author: Qi7
Date: 2022-07-13 23:30:51
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-10-29 15:56:02
Description: 
'''
import pandas as pd
import numpy as np
import torch
import matplotlib.colors
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt




class UncertaintySampling():
    """Active Learning methods to sample for uncertainty for the solvability problem.
    
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def least_confidence(self, prob):
        """
        prob: the probability output of the neural network
        """
        if prob > 0.5:
            simple_least_conf = prob
        else:
            simple_least_conf = 1 - prob
        
        #binary classification
        num_labels = 2
        normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels - 1))
        return normalized_least_conf
    
    def margin_confidence(self, prob):
        """ 
        Returns the uncertainty score of a probability distribution using
        margin of confidence sampling in a 0-1 range where 1 is the most uncertain
        """
        if prob > 0.5:
            return (1 - prob) / prob
        else:
            return prob / (1-prob)
    
    def entropy_based(self, prob):
        """
        Returns the uncertainty score of a probability distribution using
        entropy.
        """
        prob_dist = [prob, 1 - prob]
        log_probs = prob_dist * np.log2(prob_dist)
        raw_entropy = 0 - np.sum(log_probs)
        normalized_entropy = raw_entropy / np.log2(len(prob_dist))

        return normalized_entropy



def classify_sub(classifier, x_train, y_train, x_test, y_test, cm_file_name, summary_file_name, classifier_name, verbose = True):
    classifier.fit(x_train, y_train)
    """
    Machine learning methods helper functions
    """
    pred = classifier.predict(x_test)
    
    # confusion matrix
    cm = pd.crosstab(y_test, pred)
    cm.to_csv(cm_file_name)    
    
    pd.DataFrame(classification_report(y_test, pred, output_dict = True)).transpose().to_csv(summary_file_name)
    
    if verbose:
        print(classifier_name + ' Done.\n')
    
    del classifier
    del pred
    del cm


def predict(row, model):
    row = torch.Tensor([row])
    yhat = model(row)
    # Get numpy array
    yhat = yhat.detach().numpy()
    return yhat


def heatmap(model, dataset, sampled_data, device, uncertainty_methods, epoch, method, path, lb, ub, verbose, pretrained, tb=None):
    """
    Plot the 2D heatmap
    """
    preds = []
    model.eval()

    with torch.no_grad():
        X = torch.from_numpy(dataset)
            
        X = X.to(device, dtype=torch.float)
        preds = model(X)
        if tb!= None and method == 2:
            # The index specifies the 
            tb_index = (X[:,0] <= tb["x1_hi"]) & (X[:,0] >= tb["x1_lo"]) & (X[:,1] <= tb["x2_hi"]) & (X[:,1] >= tb["x2_lo"])
            preds[tb_index] = 1
        preds = preds.detach().cpu().numpy()
    
    preds = preds.flatten()
    for i in range(len(preds)):
        preds[i] = uncertainty_methods(preds[i])        
    
    feature_1 = dataset[:,0]
    feature_2 = dataset[:,1]
    
    uncertainty_threshold = 0.9
    # Return the index of samples whose uncertainty value is bigger than the mark_size_threshold
    index = np.where(preds > uncertainty_threshold)[0]
    # draw the heatmap 
    plt.figure(figsize=(19,16))
    plt.title(f"uncertainty heatmap (Epoch: {epoch})", fontsize=30)
    
    # scatter with different marker size
    plt.scatter(feature_1, feature_2, c=preds, cmap="coolwarm", vmax=1, vmin=0, s=10)
    plt.scatter(feature_1[index], feature_2[index], c=preds[index], cmap="coolwarm", vmax=1, vmin=0, s=80)
    
    plt.xlabel("feature 1", fontsize=16)
    plt.ylabel("feature 2", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=40)
    plt.scatter(sampled_data[:,0], sampled_data[:,1], c='g', marker="v", s=[1500])
    plt.plot([lb[0], ub[0]], [lb[1], lb[1]], c='y', linewidth=20)
    plt.plot([lb[0], ub[0]], [ub[1], ub[1]], c='y', linewidth=20)
    plt.plot([lb[0], lb[0]], [lb[1], ub[1]], c='y', linewidth=20)
    plt.plot([ub[0], ub[0]], [lb[1], ub[1]], c='y', linewidth=20)
    
    if method == 0:
        plot_true_bound()
    
    
    if pretrained:
        fig_name = path + f'epoch_{epoch}_transfer'
    else:
        fig_name = path + f'epoch_{epoch}'
    
    plt.savefig(fig_name, dpi=70)
    if verbose:
        plt.show()
    else:
        plt.close()
    # plt.close()
    # plt.show()
    # plt.savefig(path+f'_epoch_{epoch}',dpi=20)

def heatmap3D(model, dataset, device):
    """
    Plot the 3D heatmap
    """
    preds = []
    model.eval()
    num_of_nodes = 400

    with torch.no_grad():
        i = 0
        for X, _ in dataloader:
            i += 1
            X = X.to(device, dtype=torch.float)
            pred = model(X)
            pred = pred.cpu().numpy().item()
            preds.append(pred)
            if i == num_of_nodes:
                break

    
    feature_1 = dataset[:num_of_nodes,0]
    feature_2 = dataset[:num_of_nodes,1]
    feature_3 = dataset[:num_of_nodes,2]
    color = [plt.get_cmap("coolwarm", 100)(i) for i in preds]
    # draw the heatmap 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.set_cmap(plt.get_cmap("coolwarm", 100))
    im = ax.scatter(feature_1, feature_2, feature_3, s=100, c=color)
    fig.colorbar(im)
    ax.set_xlabel('Feature_1')
    ax.set_ylabel('Feature_2')
    ax.set_zlabel('Feature_3')

    # plt.figure()
    # plt.scatter(feature_1, feature_2, feature_3, c=preds, cmap="coolwarm")
    # plt.colorbar()
    plt.show()

def dataSampling(model, uncertainty_methods, data_pool, device):
    """
    sample the data from the data pool based on the uncertainty method. 
    return the data pool with the uncertainty from high to low.
    """
    model.eval()
    with torch.no_grad():
        X = data_pool[:, :-1]
        X = torch.from_numpy(X)
        X = X.to(device, dtype=torch.float)
        preds = model(X)
        preds = preds.detach().cpu().numpy()

    preds = preds.flatten()
    for i in range(len(preds)):
        preds[i] = uncertainty_methods(preds[i])
    
    # sort the index based on the uncertainty
    sorted_index = sorted(range(len(preds)), key=lambda k: preds[k])

    # from high to low
    return data_pool[sorted_index[::-1]]


def removeOutBound(tb, data):
    """
    Remove the data that are able to be determined by theritical bound
    """
    # data = np.delete(data, np.where(
    #     (data[:,0] >= tb["x1_hi_out"]) |
    #     (data[:,0] <= tb["x1_lo_out"]) |
    #     (data[:,1] >= tb["x2_hi_out"]) |
    #     (data[:,1] <= tb["x2_lo_out"]) |
    #     ((data[:,0] <= tb["x1_hi_in"]) & (data[:,0] >= tb["x1_lo_in"]) & (data[:,1] <= tb["x2_hi_in"]) & (data[:,1] >= tb["x2_lo_in"]))
    # )[0], axis=0)
    data = np.delete(data, np.where(
        ((data[:,0] <= tb["x1_hi"]) & (data[:,0] >= tb["x1_lo"]) & (data[:,1] <= tb["x2_hi"]) & (data[:,1] >= tb["x2_lo"]))
    )[0], axis=0)
    
    return data

def removeOutBound_4d(tb, data):
    """
    Remove the data that are able to be determined by theritical bound
    """
    # data = np.delete(data, np.where(
    #     (data[:,0] >= tb["x1_hi_out"]) |
    #     (data[:,0] <= tb["x1_lo_out"]) |
    #     (data[:,1] >= tb["x2_hi_out"]) |
    #     (data[:,1] <= tb["x2_lo_out"]) |
    #     ((data[:,0] <= tb["x1_hi_in"]) & (data[:,0] >= tb["x1_lo_in"]) & (data[:,1] <= tb["x2_hi_in"]) & (data[:,1] >= tb["x2_lo_in"]))
    # )[0], axis=0)
    data = np.delete(data, np.where(
        ((data[:,0] <= tb["x1_hi"]) & (data[:,0] >= tb["x1_lo"]) & 
         (data[:,1] <= tb["x2_hi"]) & (data[:,1] >= tb["x2_lo"]) &
         (data[:,2] <= tb["x3_hi"]) & (data[:,2] >= tb["x3_lo"]) &
         (data[:,3] <= tb["x4_hi"]) & (data[:,3] >= tb["x4_lo"]))
    )[0], axis=0)
    
    return data


def isInBound(tb, data) -> bool:
    """
    return 1 if the data is in the solvable area
    return 0 if the data is in the non-solvable area
    return False if the data can't be decided by the theoritical bound
    """
    if ((data[0] >= tb["x1_hi_out"]) or (data[0] <= tb["x1_lo_out"]) or (data[:,1] >= tb["x2_hi_out"]) or (data[:,1] <= tb["x2_lo_out"])):
        return 0
    elif ((data[0] <= tb["x1_hi_in"]) & (data[0] >= tb["x1_lo_in"]) & (data[1] <= tb["x2_hi_in"]) & (data[1] >= tb["x2_lo_in"])):
        return 1
    else:
        return False

def plot_true_bound():
    # hardcode (by sampling the position of the truth label from the plot figure) the truth boundary of the second case for visualization:
    true_x1_1, true_x2_1 = -1.265, 0.688
    true_x1_2, true_x2_2 = -0.012, 0.688
    true_x1_3, true_x2_3 = 0.342, 0.355
    true_x1_4, true_x2_4 = 0.342, -1.059
    true_x1_5, true_x2_5 = -0.781, -1.059
    true_x1_6, true_x2_6 = -1.265, -0.561
    
        # plot true boundary
    plt.plot([true_x1_1, true_x1_2], [true_x2_1, true_x2_2], c='g', linewidth=20)
    plt.plot([true_x1_2, true_x1_3], [true_x2_2, true_x2_3], c='g', linewidth=20)
    plt.plot([true_x1_3, true_x1_4], [true_x2_3, true_x2_4], c='g', linewidth=20)
    plt.plot([true_x1_4, true_x1_5], [true_x2_4, true_x2_5], c='g', linewidth=20)
    plt.plot([true_x1_5, true_x1_6], [true_x2_5, true_x2_6], c='g', linewidth=20)
    plt.plot([true_x1_6, true_x1_1], [true_x2_6, true_x2_1], c='g', linewidth=20)