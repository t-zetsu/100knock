import optuna
from gensim.models import KeyedVectors
import json
import numpy as np
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
id_dict = json.load(open('data/id_dict.json', 'r'))
train_data = pd.read_csv("data/train.feature.csv", usecols=[1,2])
valid_data = pd.read_csv("data/valid.feature.csv", usecols=[1,2])
test_data = pd.read_csv("data/test.feature.csv", usecols=[1,2])
label = {'b': 0, 't': 1, 'e':2, 'm':3}
model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
device = torch.device('cuda:0')

# 学習済み単語ベクトルの取得
VOCAB_SIZE = len(set(id_dict.values())) + 1
EMB_SIZE = 300
weights = np.zeros((VOCAB_SIZE, EMB_SIZE))
words_in_pretrained = 0
for i, word in enumerate(set(id_dict.keys())):
    if(i<VOCAB_SIZE):
        try:
            weights[i] = model[word]
            words_in_pretrained += 1
        except KeyError:
            weights[i] = np.random.normal(scale=0.4, size=(EMB_SIZE,))
    else:
        break
weights = torch.from_numpy(weights.astype((np.float32)))
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

class RNN(nn.Module):
  def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size, num_layers, emb_weights=None, bidirectional=False):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_directions = bidirectional + 1  # 単方向：1、双方向：2
    if emb_weights != None:  # 指定があれば埋め込み層の重みをemb_weightsで初期化
      self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
    else:
      self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
    self.rnn = nn.RNN(emb_size, hidden_size, num_layers, nonlinearity='tanh', bidirectional=bidirectional, batch_first=True)
    self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

  def forward(self, x):
    self.batch_size = x.size()[0]
    hidden = self.init_hidden()  # h0のゼロベクトルを作成
    emb = self.emb(x)
    # emb.size() = (batch_size, seq_len, emb_size)
    out, hidden = self.rnn(emb, hidden)
    # out.size() = (batch_size, seq_len, hidden_size * num_directions)
    out = self.fc(out[:, -1, :])
    # out.size() = (batch_size, output_size)
    return out

  def init_hidden(self):
    hidden = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size).to(device)
    return hidden
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


class Padsequence():
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x['inputs'].shape[0], reverse=True)
        sequences = [x['inputs'] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.padding_idx)
        labels = torch.LongTensor([x['labels'] for x in sorted_batch])

        return {'inputs': sequences_padded, 'labels': labels}

def tokenizer(text, id_dict=id_dict, unk=0):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return [id_dict.get(word, unk) for word in text.translate(table).split()]

def objective(trial):
  # チューニング対象パラメータのセット
    out_channels = int(trial.suggest_discrete_uniform('out_channels', 50, 200, 50))
    learning_rate = trial.suggest_loguniform('learning_rate', 5e-4, 5e-2)
    momentum = trial.suggest_discrete_uniform('momentum', 0.5, 0.9, 0.1)
    batch_size = int(trial.suggest_discrete_uniform('batch_size', 16, 128, 16))
    num_layers = int(trial.suggest_discrete_uniform('num_layers', 2, 6, 2))

  # 固定パラメータの設定
    EMB_SIZE = 300
    VOCAB_SIZE = len(set(id_dict.values())) + 1
    PADDING_IDX = len(set(id_dict.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 100
    LEARNING_RATE = trial.params['learning_rate']
    NUM_EPOCHS = 30

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE, num_layers, emb_weights=weights, bidirectional=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    device = torch.device('cuda:0')
    log = train(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=device)

    # 損失の算出
    loss_valid, _ = calculate(model, dataset_valid, device, criterion=criterion) 

    return loss_valid 


dataset_train = CreateDataset(train_data['TITLE'],train_data['CATEGORY'].replace(label),tokenizer)
dataset_valid = CreateDataset(valid_data['TITLE'],valid_data['CATEGORY'].replace(label),tokenizer)
dataset_test = CreateDataset(test_data['TITLE'],test_data['CATEGORY'].replace(label),tokenizer)

# 最適化
study = optuna.create_study()
study.optimize(objective, timeout=7200)
print('Best trial:')
trial = study.best_trial
print('  Value: {:.3f}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
  print('    {}: {}'.format(key, value))
  
VOCAB_SIZE = len(set(id_dict.values())) + 1
EMB_SIZE = 300
PADDING_IDX = len(set(id_dict.values()))
OUTPUT_SIZE = 4
HIDDEN_SIZE = 100
NUM_LAYERS = int(trial.params['num_layers'])
LEARNING_RATE = trial.params['learning_rate']
BATCH_SIZE = int(trial.params['batch_size'])
NUM_EPOCHS = 30

model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, emb_weights=weights, bidirectional=True)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
log = train(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=device)
