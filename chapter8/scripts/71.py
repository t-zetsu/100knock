import torch
import numpy as np

input_file = 'data/X_train.pt'


X_train = torch.load(input_file)
W = torch.tensor(np.ones((300, 4),np.float32))

y = torch.nn.functional.softmax(torch.matmul(X_train[0],W),0)
Y = torch.nn.functional.softmax(torch.matmul(X_train,W),1)

print(y)

print(Y)