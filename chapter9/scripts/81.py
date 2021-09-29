import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import string
device = torch.device('cpu')
print(device)

#データ読み込み
id_dict = json.load(open('data/id_dict.json', 'r'))
train_data = pd.read_csv("data/train.feature.csv", usecols=[1,2])

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden = self.init_hidden()  # h0のゼロベクトルを作成
        emb = self.emb(x)
        # emb.size() = (batch_size, seq_len, emb_size)
        out, hidden = self.rnn(emb, hidden)
        # out.size() = (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])
        # out.size() = (batch_size, output_size)
        return out
    def init_hidden(self):
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)
        return hidden

class CreateDataset(Dataset):
    def __init__(self, X, y, tokenizer):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        text = self.X[index]
        inputs = self.tokenizer(text)

        return {
          'inputs': torch.tensor(inputs, dtype=torch.int64),
          'labels': torch.tensor(self.y[index], dtype=torch.int64)
        }

def tokenizer(text, id_dict=id_dict, unk=0):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return [id_dict.get(word, unk) for word in text.translate(table).split()]

#データセット作成
train = CreateDataset(train_data['TITLE'],train_data['CATEGORY'].replace( {'b':0,'t':1,'e':2,'m':3}),tokenizer)

#パラメータ
VOCAB_SIZE = len(set(id_dict.values())) + 1
EMB_SIZE = 300
PADDING_IDX = len(set(id_dict.values()))
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50


model = RNN(VOCAB_SIZE,EMB_SIZE,PADDING_IDX,OUTPUT_SIZE,HIDDEN_SIZE).to(device)

for i in range(5):
    print(f'データ{i}の予測分布')
    print(torch.softmax(model(train[i]['inputs'].unsqueeze(0)),-1))