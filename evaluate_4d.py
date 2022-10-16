'''
Author: Qi7
Date: 2022-08-03 19:48:05
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-10-16 16:42:00
Description: this python file is for plotting the history figure such as the loss and accuracy, F1 score
'''
from cProfile import label
import numpy as np
from matplotlib import pyplot as plt

# random timesteamp history
r_t1 = np.load("savedModel/random_sample/4d_1_epochs200_lr_0.001_bs_16_history.npy", allow_pickle=True).item()['f1_test']
r_t2 = np.load("savedModel/random_sample/4d_2_epochs200_lr_0.001_bs_16_history.npy", allow_pickle=True).item()['f1_test']
r_t3 = np.load("savedModel/random_sample/4d_3_epochs200_lr_0.001_bs_16_history.npy", allow_pickle=True).item()['f1_test']
r_t4 = np.load("savedModel/random_sample/4d_4_epochs200_lr_0.001_bs_16_history.npy", allow_pickle=True).item()['f1_test']
r_t5 = np.load("savedModel/random_sample/4d_5_epochs200_lr_0.001_bs_16_history.npy", allow_pickle=True).item()['f1_test']
r_t6 = np.load("savedModel/random_sample/4d_6_epochs200_lr_0.001_bs_16_history.npy", allow_pickle=True).item()['f1_test']
r_t7 = np.load("savedModel/random_sample/4d_7_epochs200_lr_0.001_bs_16_history.npy", allow_pickle=True).item()['f1_test']

random_f1_test = r_t1 + r_t2 + r_t3 + r_t4 + r_t5 + r_t6 + r_t7

# active learning timestamp history
a_t1 = np.load("savedModel/theoretical/4d_1_epochs200_lr_0.001_bs_16_history.npy", allow_pickle=True).item()['f1_test']
a_t2 = np.load("savedModel/theoretical/4d_2_epochs200_lr_0.001_bs_16_history_transfer.npy", allow_pickle=True).item()['f1_test']
a_t3 = np.load("savedModel/theoretical/4d_3_epochs200_lr_0.001_bs_16_history_transfer.npy", allow_pickle=True).item()['f1_test']
a_t4 = np.load("savedModel/theoretical/4d_4_epochs200_lr_0.001_bs_16_history_transfer.npy", allow_pickle=True).item()['f1_test']
a_t5 = np.load("savedModel/theoretical/4d_5_epochs200_lr_0.001_bs_16_history_transfer.npy", allow_pickle=True).item()['f1_test']
a_t6 = np.load("savedModel/theoretical/4d_6_epochs200_lr_0.001_bs_16_history_transfer.npy", allow_pickle=True).item()['f1_test']
a_t7 = np.load("savedModel/theoretical/4d_7_epochs200_lr_0.001_bs_16_history_transfer.npy", allow_pickle=True).item()['f1_test']

active_f1_test = a_t1 + a_t2 + a_t3 + a_t4 + a_t5 + a_t6 + a_t7

plt.figure(figsize=(20, 14))
# plt.title('Test F1 score comparison between proposed method and conventional method', fontsize=30)
plt.plot(random_f1_test, label="conventional method", linewidth=5)
plt.plot(active_f1_test, label="proposed method", linewidth=5)
plt.xlabel("Epoch number", fontsize=25)
plt.ylabel("F1 Score", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlim([0,1400])
plt.ylim([0, 1])
plt.legend(fontsize=25)
plt.grid()
plt.show()

# random_history = np.load("savedModel/random_sample/4d_7_epochs200_lr_0.001_bs_16_history.npy", allow_pickle=True).item()
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

# plt.figure(figsize=(10,8))
# plt.title('Test accuracy')
# plt.plot(random_history['acc_test'], label="random sample")
# plt.plot(active_history['acc_test'], label="active learning sample")
# plt.plot(theoretical_history['acc_test'], label="theoretical bound sample")
# # plt.plot(transfer_history['acc_test'], label="transfer_history sample")
# plt.legend()
# plt.show()

# plt.figure(figsize=(10,8))
# plt.title('Test F1', fontsize=30)
# plt.plot(random_history['f1_test'], label="Conventional training", linewidth=5)
# plt.plot(active_history['f1_test'], label="Proposed", linewidth=5)
# plt.xlabel("Epochs", fontsize=16)
# plt.ylabel("F1 score", fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.ylim([0, 1])
# plt.plot(theoretical_history['f1_test'], label="theoretical bound sample")
# plt.plot(transfer_history['f1_test'], label="transfer_historysample")
# plt.legend(fontsize=25)
# plt.show()

