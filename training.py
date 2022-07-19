'''
Author: Qi7
Date: 2022-07-19 08:31:52
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-07-19 09:46:01
Description: 
'''
from numpy import dtype
import torch 
from tqdm import tqdm

def get_accuracy(pred, label):
    correct, total = 0, 0
    pred = pred.max(1, keepdim=True)[1]
    correct += pred.eq(label.view_as(pred)).sum().item()
    total += int(label.shape[0])
    return correct / total


def train_loop(trainLoader, model, device, LR, metric_fn, loss_fn, history, is_pretrained = False):
    num_batches = len(trainLoader)
    if is_pretrained == False:
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    else:
        # frozen the layer no need to train
        # for weights in model.encoder.parameters():
        #     weights.requires_grad = False
        
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.SGD(params, lr=LR)
    
    model.train()
    train_loss_sum = 0
    total = 0
    data_size = len(trainLoader.dataset)
    for batch, (data_batch, labels) in tqdm(enumerate(trainLoader)):
        # data_batch = data_batch.view(data_batch.shape[0], 1, data_batch.shape[1])
        data_batch = data_batch.to(device, dtype=torch.float)
        labels = labels.view(labels.shape[0], 1)
        labels = labels.to(device, dtype=torch.float)
        pred = model(data_batch)
        loss = loss_fn(pred, labels)
        metric = metric_fn(pred, labels)
        
        # if don't call zero_grad, the grad of each batch will be accumulated
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(data_batch)
            # metric = metric.item()
            print(f"loss:{loss:>7f})----- metric: {metric:>7f}    [{current:>5d}/{data_size:>5d}]")
            history['train'].append(loss)
            history['f1_train'].append(metric)

    # Here maybe add the accuracy
    print(f"F1 score accuracy:{metric}")

def eval_loop(dataloader, model, epoch, loss_fn, metric_fn, device, history):
    num_batches = len(dataloader)
    eval_metric = 0
    loss = 0
    cnt = 0
    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            # X = X.view(X.shape[0], 1, X.shape[1])
            X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.float)
            pred = model(X)
            loss += loss_fn(pred, y)
            eval_metric += metric_fn(pred, y)
        loss /= num_batches

        # Here maybe the accuracy
        eval_metric /= num_batches
    print(f"Test accuracy: {eval_metric:>8f} \n")
    print(f"Test loss: {loss:>8f} \n")
    history['val'].append(loss.item())
    history['f1_test'].append(eval_metric)