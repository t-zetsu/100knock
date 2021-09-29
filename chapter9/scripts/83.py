import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import string
import time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

id_dict = json.load(open('data/id_dict.json', 'r'))
train_data = pd.read_csv("data/train.feature.csv", usecols=[1,2])
valid_data = pd.read_csv("data/valid.feature.csv", usecols=[1,2])
test_data = pd.read_csv("data/test.feature.csv", usecols=[1,2])
label = {'b': 0, 't': 1, 'e':2, 'm':3}

def tokenizer(text, id_dict=id_dict, unk=0):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return [id_dict.get(word, unk) for word in text.translate(table).split()]

class Padsequence():
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x['inputs'].shape[0], reverse=True)
        sequences = [x['inputs'] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.padding_idx)
        labels = torch.LongTensor([x['labels'] for x in sorted_batch])

        return {'inputs': sequences_padded, 'labels': labels}

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
        hidden = torch.zeros(1, self.batch_size, self.hidden_size).to(device)
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

def calculate(model, dataset, device=None, criterion=None): #datasetの平均lossと平均accuracyを計算
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)

            outputs = model(inputs)

            if criterion != None:
                loss += criterion(outputs, labels).item()
                
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

    return loss / len(dataset), correct / total

def train(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, device=None, collate_fn=None):
    model.to(device)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5, last_epoch=-1)
    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        s_time = time.time()
        model.train()
        for data in dataloader_train:
            optimizer.zero_grad()
            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        loss_train, acc_train = calculate(model, dataset_train, device,criterion)
        loss_valid, acc_valid = calculate(model,dataset_valid, device, criterion)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])
        
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoints/checkpoint{epoch + 1}.pt')

        e_time = time.time()

        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec') 

        if epoch > 2 and log_valid[epoch - 3][0] <= log_valid[epoch - 2][0] <= log_valid[epoch - 1][0] <= log_valid[epoch][0]:
            break

        scheduler.step()

    return {'train': log_train, 'valid': log_valid}

#Tokenize
dataset_train = CreateDataset(train_data['TITLE'],train_data['CATEGORY'].replace(label),tokenizer)
dataset_valid = CreateDataset(valid_data['TITLE'],valid_data['CATEGORY'].replace(label),tokenizer)
dataset_test = CreateDataset(test_data['TITLE'],test_data['CATEGORY'].replace(label),tokenizer)

VOCAB_SIZE = len(set(id_dict.values())) + 1 
EMB_SIZE = 300
PADDING_IDX = len(set(id_dict.values()))
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
NUM_EPOCHS = 10

# モデルの定義
model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# オプティマイザの定義
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# モデルの学習
log = train(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=device)