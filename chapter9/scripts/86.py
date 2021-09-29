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

class CNN(nn.Module):
  def __init__(self, vocab_size, emb_size, padding_idx, output_size, out_channels, kernel_heights, stride, padding, emb_weights=None):
    super().__init__()
    if emb_weights != None: 
      self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
    else:
      self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
    self.conv = nn.Conv2d(1, out_channels, (kernel_heights, emb_size), stride, (padding, 0))
    self.drop = nn.Dropout(0.3)
    self.fc = nn.Linear(out_channels, output_size)

  def forward(self, x):
    emb = self.emb(x).unsqueeze(1)
    conv = self.conv(emb)
    act = F.relu(conv.squeeze(3))
    max_pool = F.max_pool1d(act, act.size()[2])
    out = self.fc(self.drop(max_pool.squeeze(2)))
    return out

def tokenizer(text, id_dict=id_dict, unk=0):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return [id_dict.get(word, unk) for word in text.translate(table).split()]

dataset_train = CreateDataset(train_data['TITLE'],train_data['CATEGORY'].replace(label),tokenizer)
dataset_valid = CreateDataset(valid_data['TITLE'],valid_data['CATEGORY'].replace(label),tokenizer)
dataset_test = CreateDataset(test_data['TITLE'],test_data['CATEGORY'].replace(label),tokenizer)

VOCAB_SIZE = len(set(id_dict.values())) + 1
EMB_SIZE = 300
PADDING_IDX = len(set(id_dict.values()))
OUTPUT_SIZE = 4
OUT_CHANNELS = 100
KERNEL_HEIGHTS = 3
STRIDE = 1
PADDING = 1

model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=weights)

X = dataset_train[0]['inputs']
print(torch.softmax(model(X.unsqueeze(0)), dim=-1))