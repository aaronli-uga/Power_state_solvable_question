'''
Author: Qi7
Date: 2022-07-19 08:23:59
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-07-21 20:49:40
Description: 
'''
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder # The LabelEncoder turns catergorial features into numerical features.

class MyLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
        self.length = self.data.shape[0]
    
    def __getitem__(self,index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    
    def __len__(self):
        return len(self.data)