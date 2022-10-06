'''
Author: Qi7
Date: 2022-08-03 19:48:05
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-10-04 22:53:48
Description: this python file is for plotting the history figure such as the loss and accuracy, F1 score
'''
from cProfile import label
import numpy as np
from matplotlib import pyplot as plt

random_history = np.load("savedModel/random_sample/4d_7_epochs200_lr_0.001_bs_16_history.npy", allow_pickle=True).item()
# active_history= np.load("savedModel/active_learning/4d_1_epochs200_lr_0.001_bs_16_history.npy", allow_pickle=True).item()
# theoretical_history = np.load("savedModel/theoretical/v2_epochs200_lr_0.001_bs_16_history.npy", allow_pickle=True).item()
# transfer_history= np.load("savedModel/theoretical/v2_epochs200_lr_0.001_bs_16_history _transfer.npy", allow_pickle=True).item()

# plt.figure(figsize=(10,8))
# plt.title('Test Loss', fontsize=30)
# plt.plot(random_history['test_loss'], label="Conventional training")
# # plt.plot(active_history['test_loss'], label="Proposed")
# plt.xlabel("Epochs", fontsize=16)
# plt.ylabel("Loss", fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# # plt.plot(theoretical_history['test_loss'], label="theoretical bound sample")
# # plt.plot(transfer_history['test_loss'], label="transfer_history sample")
# plt.legend(fontsize=25)
# plt.show()

plt.figure(figsize=(10,8))
plt.title('Test accuracy')
plt.plot(random_history['acc_test'], label="random sample")
# plt.plot(active_history['acc_test'], label="active learning sample")
# plt.plot(theoretical_history['acc_test'], label="theoretical bound sample")
# # plt.plot(transfer_history['acc_test'], label="transfer_history sample")
plt.legend()
plt.show()

plt.figure(figsize=(10,8))
plt.title('Test F1', fontsize=30)
plt.plot(random_history['f1_test'], label="Conventional training", linewidth=5)
# plt.plot(active_history['f1_test'], label="Proposed", linewidth=5)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("F1 score", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim([0, 1])
# plt.plot(theoretical_history['f1_test'], label="theoretical bound sample")
# plt.plot(transfer_history['f1_test'], label="transfer_historysample")
plt.legend(fontsize=25)
plt.show()

