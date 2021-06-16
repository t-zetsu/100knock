from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

# データのロード
x_train = pd.read_csv('/work/chapter5/data/train.feature.txt', sep='\t', header=None)
y_train = pd.read_csv('/work/chapter5/data/train.txt', sep='\t', header=None)[0]
# クラスを置換
category = {"b":0, "t":1, "e":2, "m":3}
y_train = y_train.replace(category)
# 学習
model = LogisticRegression(max_iter=200, random_state=0)
model.fit(x_train, y_train)
# モデルの保存
filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))