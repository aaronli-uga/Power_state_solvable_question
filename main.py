'''
Author: Qi7
Date: 2022-07-13 22:39:08
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-07-19 00:50:16
Description: 
'''
#%% Loading data from csv
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from utils import classify_sub
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

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

# print chart plot
cnt = Counter(y)
data = {'Solvable':cnt[0], 'Non-Solvable':cnt[1]}
data_type = list(data.keys())
data_numbers = list(data.values())

fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(data_type, data_numbers, color ='blue',
        width = 0.4)
 
plt.xlabel("data types ")
plt.ylabel("No. of samples")
plt.title("data distribution")
plt.show()

#%%
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(X_train[:,0], X_train[:,1], c=y_train.flatten())
ax.grid(True)
plt.show()


# %% Machine learning training
confusion_matrix_folder = "Confusion_Matrix/"
summary_folder = "Summary/"
prefix = "LogisticRegression"

if os.path.isdir(confusion_matrix_folder) == False:
        os.mkdir(confusion_matrix_folder)
if os.path.isdir(summary_folder) == False:
        os.mkdir(summary_folder)

linear_classifier = LogisticRegression(random_state = 42)
classify_sub(linear_classifier, 
                X_train, y_train,
                X_test, y_test, 
                confusion_matrix_folder + prefix + '_cm_linear.csv', 
                summary_folder + prefix + '_summary_linear.csv',
                'Linear',
                verbose=True)

prefix = "SVM_linear"
svm_classifier = LinearSVC(random_state = 42)
classify_sub(svm_classifier, 
                X_train, y_train, 
                X_test, y_test, 
                confusion_matrix_folder + prefix + '_cm_svm.csv', 
                summary_folder + prefix + '_summary_svm.csv',
                'SVM',
                verbose=True)

prefix = "SVM_rbf"
kernel_svm_classifier = SVC(kernel = 'rbf', random_state = 42, gamma='scale')
classify_sub(kernel_svm_classifier, 
                X_train, y_train, 
                X_test, y_test, 
                confusion_matrix_folder + prefix + '_cm_kernel_svm.csv', 
                summary_folder + prefix + '_summary_kernel_svm.csv',
                'SVM',
                verbose=True)