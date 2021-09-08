import torch
import numpy as np
import random
import math

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

def adam(W,grad,m,v):
    a=0.001;b1=0.9;b2=0.999;e=0.00000001
    m = b1*m+(1-b1)*grad
    v = b2*v+(1-b2)*grad*grad
    m_ = m/(1-b1)
    v_ = v/(1-b2)
    W = W - a*m_/(torch.sqrt(v_)+e)
    return W, m, v

def SGD(X_train,X_valid,y_train,y_valid,W):
    m=0;v=0
    for epoch in range(n_epoch):
        shuffle_list = random.sample([i for i in range(X_train.size()[0])],X_train.size()[0]) #訓練データシャッフルのためのリスト
        for j in range(X_train.size()[0]): 
            X = X_train[shuffle_list[j]:shuffle_list[j]+1] #データ１つずつ取り出す
            y = y_train[shuffle_list[j]:shuffle_list[j]+1]
            W, m, v = adam(W,torch.t(gradient(X,y,W)),m,v) #重み更新
        print(f'epoch {epoch+1} : loss_train {cross_entropy(X_train,y_train,W)}   loss_valid {cross_entropy(X_valid,y_valid,W)}')
    return W

X_train = torch.load('data/X_train.pt')
y_train = torch.load('data/y_train.pt')
X_valid = torch.load('data/X_valid.pt')
y_valid = torch.load('data/y_valid.pt')
X_test = torch.load('data/X_test.pt')
y_test = torch.load('data/y_test.pt')
output_file = 'data/model.pt'

n_label = 4
n_dim = 300
n_epoch = 30
W = SGD(X_train,X_valid,y_train,y_valid,torch.zeros(n_dim, n_label, dtype=torch.float))
torch.save(W,output_file)