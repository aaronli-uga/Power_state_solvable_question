'''
Author: Qi7
Date: 2022-08-03 19:48:05
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-08-05 18:11:07
Description: 
'''
from cProfile import label
import numpy as np
from matplotlib import pyplot as plt

random_history = np.load("savedModel/random_sample/epochs200_lr_0.001_bs_16_history.npy", allow_pickle=True).item()
active_history= np.load("savedModel/active_learning/epochs200_lr_0.001_bs_16_history.npy", allow_pickle=True).item()
theoretical_history = np.load("savedModel/theoretical/epochs200_lr_0.001_bs_16_history.npy", allow_pickle=True).item()

plt.figure(figsize=(10,8))
plt.title('Test Loss')
plt.plot(random_history['test_loss'], label="random sample")
plt.plot(active_history['test_loss'], label="active learning sample")
plt.plot(theoretical_history['test_loss'], label="theoretical bound sample")
plt.legend()
plt.show()

plt.figure(figsize=(10,8))
plt.title('Test accuracy')
plt.plot(random_history['acc_test'], label="random sample")
plt.plot(active_history['acc_test'], label="active learning sample")
plt.plot(theoretical_history['acc_test'], label="theoretical bound sample")
plt.legend()
plt.show()

plt.figure(figsize=(10,8))
plt.title('Test F1')
plt.plot(random_history['f1_test'], label="random sample")
plt.plot(active_history['f1_test'], label="active learning sample")
plt.plot(theoretical_history['f1_test'], label="theoretical bound sample")
plt.legend()
plt.show()