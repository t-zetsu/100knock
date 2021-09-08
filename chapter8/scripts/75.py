import torch
import numpy as np
import random
import math
from matplotlib import pyplot as plt

def cross_entropy(X,y,W):
    P = torch.nn.functional.softmax(torch.matmul(X,W),1) #予測行列
    P_list = P.tolist()
    y_list = y.tolist()
    loss_list = [-math.log(P_list[i][y[i]]) for i in range(len(P_list))]
    loss = sum(loss_list)/len(loss_list)
    return loss

def gradient(X,y,W):
    grad = torch.tensor([[0 for j in range(n_dim)] for i in range(n_label)],dtype=float) #勾配の初期化
    y_list = y.tolist()
    n = len(y_list)
    P = torch.nn.functional.softmax(torch.matmul(X,W),1) #予測行列
    T = torch.tensor([[1 if j == y_list[i] else 0 for j in range(n_label)] for i in range(n)]) #正解行列
    for i in range(n):
        grad += torch.matmul(torch.t((P - T)[i:i+1]),X[i:i+1]) #各事例に対する勾配の和を求める
    grad = grad/n #平均
    return grad.float()

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

def adam(W,grad,m,v):
    a=0.001;b1=0.9;b2=0.999;e=0.00000001
    m = b1*m+(1-b1)*grad
    v = b2*v+(1-b2)*grad*grad
    m_ = m/(1-b1)
    v_ = v/(1-b2)
    W = W - a*m_/(torch.sqrt(v_)+e)
    return W, m, v

def plot_log(log_train,log_valid):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.array(log_train).T[0], label='train')
    ax[0].plot(np.array(log_valid).T[0], label='valid')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].legend()
    ax[1].plot(np.array(log_train).T[1], label='train')
    ax[1].plot(np.array(log_valid).T[1], label='valid')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('accuracy')
    ax[1].legend()
    fig.savefig(output_file)

def SGD(X_train,X_valid,y_train,y_valid,W):
    m=0;v=0
    log_train=[];log_valid=[]
    for epoch in range(n_epoch):
        shuffle_list = random.sample([i for i in range(X_train.size()[0])],X_train.size()[0]) #訓練データシャッフルのためのリスト
        for j in range(X_train.size()[0]): 
            X = X_train[shuffle_list[j]:shuffle_list[j]+1] #データ１つずつ取り出す
            y = y_train[shuffle_list[j]:shuffle_list[j]+1]
            W, m, v = adam(W,torch.t(gradient(X,y,W)),m,v) #重み更新
        loss_train = cross_entropy(X_train,y_train,W)
        loss_valid = cross_entropy(X_valid,y_valid,W)
        accuracy_train = accuracy(X_train,y_train,W)
        accuracy_valid = accuracy(X_valid,y_valid,W)
        log_train.append([loss_train, accuracy_train])
        log_valid.append([loss_valid, accuracy_valid])
        print(f'epoch {epoch+1} : loss_train {loss_train}, accuracy_train {accuracy_train}, loss_valid {loss_valid}, accuracy_valid {accuracy_valid}')
    plot_log(log_train,log_valid)
    return W

X_train = torch.load('data/X_train.pt')
y_train = torch.load('data/y_train.pt')
X_valid = torch.load('data/X_valid.pt')
y_valid = torch.load('data/y_valid.pt')
output_file = 'outputs/75.png'

n_label = 4
n_dim = 300
n_epoch = 30
W = SGD(X_train,X_valid,y_train,y_valid,W = torch.zeros(n_dim, n_label, dtype=torch.float))