import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
import string
import torch

model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True) 

def text_to_vec(text):
  words = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).split() 
  vec = [model[word] for word in words if word in model]

  return torch.tensor(sum(vec) / len(vec))

label = {'b': 0, 't': 1, 'e':2, 'm':3}
train = pd.read_csv('data/50.train.csv')
valid = pd.read_csv('data/50.valid.csv')
test = pd.read_csv('data/50.test.csv')

X_train = torch.stack([text_to_vec(text) for text in train['TITLE']])
X_valid = torch.stack([text_to_vec(text) for text in valid['TITLE']])
X_test = torch.stack([text_to_vec(text) for text in test['TITLE']])

y_train = torch.tensor(train['CATEGORY'].map(lambda x: label[x]).values)
y_valid = torch.tensor(valid['CATEGORY'].map(lambda x: label[x]).values)
y_test = torch.tensor(test['CATEGORY'].map(lambda x: label[x]).values)


torch.save(X_train, 'data/X_train.pt')
torch.save(X_valid, 'data/X_valid.pt')
torch.save(X_test, 'data/X_test.pt')
torch.save(y_train, 'data/y_train.pt')
torch.save(y_valid, 'data/y_valid.pt')
torch.save(y_test, 'data/y_test.pt')