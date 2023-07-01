'''
Author: Qi7
Date: 2023-05-08 00:11:16
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-22 14:22:47
Description: 
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

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')

X_test = np.load("X_test.npy")
y_test = np.load("Y_test.npy")
# Adding noise
noise_levels = [0.03, 0.1, 0.2, 0.3, 0.4]
history = dict(train_loss=[], test_loss=[], acc_train=[], acc_test=[], f1_train=[], f1_test=[])

for noise_level in noise_levels:
    
    noise_feature1 = np.random.normal(0, X_test[:,0].std(), X_test.shape[0]) * noise_level
    noise_feature2 = np.random.normal(0, X_test[:,1].std(), X_test.shape[0]) * noise_level
    X_test[:,0] = X_test[:,0] + noise_feature1
    X_test[:,1] = X_test[:,1] + noise_feature2

    trained_model = "savedModel/active_learning/2d_epochs120_lr_0.001_bs_16_bestmodel.pth"
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
    

print(history['f1_test'])
print(history['acc_test'])

for i in range(len(history['acc_test'])):
    history['acc_test'][i] = round(history['acc_test'][i], 3)

objects = ('3%', '10%', '20%', '30%', '40%')
y_pos = np.arange(len(objects))

plt.bar(y_pos, history['acc_test'], align='center', alpha=0.5)
addlabels(y_pos, history['acc_test'])
plt.xticks(y_pos, objects)
plt.ylabel('Test Accuracy')
plt.xlabel('Noise levels in percentage')
plt.title('Accuracy in different levels of Gaussion noise')

plt.show()

