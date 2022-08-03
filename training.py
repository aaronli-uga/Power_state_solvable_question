'''
Author: Qi7
Date: 2022-07-19 08:31:52
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-08-03 11:29:15
Description: 
'''
from curses import mousemask
from numpy import dtype
import torch 
import math
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score

def train_loop(trainLoader, model, device, LR, metric_fn, loss_fn, history, save_path = 'solvable_best_model.pth', is_pretrained = False, momentum=0.9, verbose=False):
    num_batches = len(trainLoader)
    if is_pretrained == False:
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=momentum)
        # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    else:
        # frozen the layer no need to train
        # for weights in model.encoder.parameters():
        #     weights.requires_grad = False
        
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params, lr=LR)
        # optimizer = torch.optim.Adam(params, lr=LR)

    
    model.train()
    data_size = len(trainLoader.dataset)
    for batch, (data_batch, labels) in enumerate(trainLoader):
        # data_batch = data_batch.view(data_batch.shape[0], 1, data_batch.shape[1])
        data_batch = data_batch.to(device, dtype=torch.float)
        # labels = labels.view(labels.shape[0], 1)
        labels = labels.to(device, dtype=torch.float)
        pred = model(data_batch)
        loss = loss_fn(pred, labels)
        # metric = metric_fn(pred, labels)
        
        # if don't call zero_grad, the grad of each batch will be accumulated
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(data_batch)
            metric = metric_fn(pred.round().cpu().detach().numpy(), labels.cpu().detach().numpy())
            # metric = metric.item()
            if verbose:
                print(f"loss:{loss:>7f})----- metric: {metric:>7f}    [{current:>5d}/{data_size:>5d}]")
            # history['train'].append(loss)
            # history['f1_train'].append(metric)
    
    print(f"Loss: {loss:>3f}, Metric: {metric:>3f}")
    history['train_loss'].append(loss.item())
    history['acc_train'].append(metric)
    torch.save(model, save_path)

def eval_loop(dataloader, model, epoch, loss_fn, metric_fn, device, history, beta=1.0, verbose=False):
    num_batches = len(dataloader)
    eval_metric = 0
    loss = 0
    cnt = 0
    preds = []
    actuals = []
    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X, y= X.to(device, dtype=torch.float), y.to(device, dtype=torch.float)
            pred = model(X)
            loss += loss_fn(pred, y)
            pred = pred.cpu().numpy()
            y = y.cpu().numpy()
            pred = pred.round()
            preds.append(pred)
            actuals.append(y)
            # eval_metric += metric_fn(pred, y)
        loss /= num_batches
        
        preds, actuals = np.vstack(preds), np.vstack(actuals)
        #Calculate metrics
        cm = confusion_matrix(actuals, preds)
        # Get descriptions of tp, tn, fp, fn
        tn, fp, fn, tp = cm.ravel()
        total = sum(cm.ravel())
        metrics = {
            'accuracy': accuracy_score(actuals, preds),
            'AU_ROC': roc_auc_score(actuals, preds),
            'f1_score': f1_score(actuals, preds),
            'average_precision_score': average_precision_score(actuals, preds),
            'f_beta': ((1+beta**2) * precision_score(actuals, preds) * recall_score(actuals, preds)) / (beta**2 * precision_score(actuals, preds) + recall_score(actuals, preds)),
            'matthews_correlation_coefficient': (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),
            'precision': precision_score(actuals, preds),
            'recall': recall_score(actuals, preds),
            'true_positive_rate_TPR':recall_score(actuals, preds),
            'false_positive_rate_FPR':fp / (fp + tn) ,
            'false_discovery_rate': fp / (fp +tp),
            'false_negative_rate': fn / (fn + tp) ,
            'negative_predictive_value': tn / (tn+fn),
            'misclassification_error_rate': (fp+fn)/total ,
            'sensitivity': tp / (tp + fn),
            'specificity': tn / (tn + fp),
            #'confusion_matrix': confusion_matrix(actuals, preds), 
            'TP': tp,
            'FP': fp, 
            'FN': fn, 
            'TN': tn
        }
        # Here maybe the accuracy
        # eval_metric /= num_batches
    print(f"\nTest Accuracy: {metrics['accuracy']:>5f}")
    print(f"Test F1 Score: {metrics['f1_score']:>5f}")
    print(f"Test Loss: {loss:>5f} \n")
    print(cm)
    history['test_loss'].append(loss.item())
    history['f1_test'].append(metrics['f1_score'])
    history['acc_test'].append(metrics['accuracy'])