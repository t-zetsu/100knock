from collections import defaultdict
import string
import pandas as pd
import json

def mkdict(x):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    for word in x.translate(table).split():
        d[word]+=1

def tokenizer(text, id_dict, unk=0):
  table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
  return [id_dict.get(word, unk) for word in text.translate(table).split()]

train = pd.read_csv("data/train.feature.csv", usecols=[1])
d = defaultdict(int)
train["TITLE"].apply(mkdict)
d = sorted(d.items(), key=lambda x:x[1], reverse=True)

def main():
    #id辞書
    id_dict = dict()
    i=1
    for w, n in d:
        if n>=2:
            id_dict[w]=i
            i+=1
        else:
            id_dict[w]=0
    #辞書の保存
    tf = open("data/id_dict.json", "w")
    json.dump(id_dict,tf)
    tf.close()
    #Tokenize
    text = train.iloc[1, train.columns.get_loc('TITLE')]
    print(f'テキスト: {text}')
    print(f'ID列: {tokenizer(text,id_dict)}')

main()
