import torch
import numpy as np
import random
import math

def accuracy(X,y,W):
    P = torch.nn.functional.softmax(torch.matmul(X,W),1)
    values, pred = torch.max(P,1)
    pred_list = pred.tolist()
    y_list = y.tolist()
    acc=0;n=len(pred_list)
    for i in range(n):
        if pred_list[i] == y_list[i]:
            acc += 1
    return acc/n

X_train = torch.load('data/X_train.pt')
y_train = torch.load('data/y_train.pt')
X_valid = torch.load('data/X_valid.pt')
y_valid = torch.load('data/y_valid.pt')
W = torch.load('data/model.pt')

print(f'学習データの正解率：{accuracy(X_train,y_train,W)}')
print(f'検証データの正解率：{accuracy(X_valid,y_valid,W)}')