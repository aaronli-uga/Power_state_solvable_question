'''
Author: Qi7
Date: 2022-07-13 23:30:51
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-07-21 23:49:23
Description: 
'''
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report

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