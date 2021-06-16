from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import pickle
import numpy as np
import pprint

# モデルのロード
filename = "/work/chapter5/model/model.sav"
model = pickle.load(open(filename, 'rb'))

category = {0:"b", 1:"t", 2:"e", 3:"m"}
x_train = pd.read_csv("/work/chapter5/data/train.feature.txt", sep='\t')
x_train.replace(category, inplace=True)

def main():
    #カラム名
    features = x_train.columns.values
    #重みの高い特徴量トップ10
    print("----------High weighted features---------")
    for c,coef in zip(model.classes_, model.coef_):
        idx = np.argsort(coef)
        print(category[c],features[idx][-10:][::-1])
    #重みの低い特徴量ワースト10
    print("----------Low weighted features----------")
    for c,coef in zip(model.classes_, model.coef_):
        idx = np.argsort(coef)
        print(category[c],features[idx][:10])

main()
