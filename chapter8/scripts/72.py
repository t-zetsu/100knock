import torch
import numpy as np
import random
import math

X_train = torch.load('data/X_train.pt')
y_train = torch.load('data/y_train.pt')
n_label = 4
n_dim = 300
W = torch.tensor([[random.random() for j in range(n_label)] for i in range(n_dim)])

def cross_entropy(y,P):
    P_list = P.tolist()
    y_list = y.tolist()
    loss_list = [-math.log(P_list[i][y[i]]) for i in range(len(P_list))]
    loss = sum(loss_list)/len(loss_list)
    return loss

def gradient(X, Y, P):
    grad = torch.tensor([[0 for j in range(n_dim)] for i in range(n_label)],dtype=float) #勾配の初期化
    Y_list = Y.tolist()
    n = len(Y_list)
    T = torch.tensor([[1 if j == Y_list[i] else 0 for j in range(n_label)] for i in range(n)]) #正解行列
    for i in range(n):
        grad += torch.matmul(torch.t((P - T)[i:i+1]),X[i:i+1]) #各事例に対する勾配の和を求める
    grad = grad/n #平均
    return grad

x_1 = X_train[:1]
X = X_train[:4]
y_1 = y_train[:1]
Y = y_train[:4]
p_1 = torch.nn.functional.softmax(torch.matmul(X_train[:1],W),1)
P = torch.nn.functional.softmax(torch.matmul(X_train[:4],W),1)

print('一つの事例')
print(f'クロスエントロピー：{cross_entropy(y_1,p_1)}')
print(f'勾配：{gradient(x_1,y_1,p_1)}')
print('複数の事例')
print(f'クロスエントロピー：{cross_entropy(Y,P)}')
print(f'勾配：{gradient(X,Y,P)}')