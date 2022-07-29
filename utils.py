'''
Author: Qi7
Date: 2022-07-13 23:30:51
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-07-29 09:58:46
Description: 
'''
import pandas as pd
import numpy as np
import torch
import matplotlib.colors
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt



def classify_sub(classifier, x_train, y_train, x_test, y_test, cm_file_name, summary_file_name, classifier_name, verbose = True):
    classifier.fit(x_train, y_train)
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


def heatmap(model, dataset, dataloader, device):
    preds = []
    model.eval()

    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device, dtype=torch.float)
            pred = model(X)
            pred = pred.cpu().numpy().item()
            preds.append(pred)
    
    feature_1 = dataset[:,0]
    feature_2 = dataset[:,1]
    # draw the heatmap 
    plt.figure()
    plt.scatter(feature_1, feature_2, c=preds, cmap="coolwarm")
    plt.colorbar()
    plt.show()

def heatmap3D(model, dataset, dataloader, device):
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