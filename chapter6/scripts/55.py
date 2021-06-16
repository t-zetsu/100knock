from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle
import numpy as np

# モデルのロード
filename = "/work/chapter5/model/model.sav"
model = pickle.load(open(filename, 'rb'))

# データのロード
x_train = pd.read_csv("/work/chapter5/data/train.feature.txt", sep='\t', header=None)
x_test = pd.read_csv('/work/chapter5/data/test.feature.txt', sep='\t', header=None)
y_train = pd.read_csv('/work/chapter5/data/train.txt', sep='\t', header=None)[0]
y_test = pd.read_csv('/work/chapter5/data/test.txt', sep='\t', header=None)[0]

# 予測結果とその確率を表示
def predict(model, data):
    prob = pd.DataFrame(np.max(model.predict_proba(data), axis=1))
    prob.columns = ["Probability"]
    pred = pd.DataFrame(model.predict(data))
    pred.columns = ["Predict"]
    result = pd.concat([prob, pred], axis = 1)
    return result


def main():
    category = {0:"b", 1:"t", 2:"e", 3:"m"}
    # 予測
    result_train = predict(model, x_train)
    result_test = predict(model, x_test)
    result_train.replace(category, inplace=True)
    result_test.replace(category, inplace=True)
    # リスト化
    pred_train_list = result_train["Predict"].values.tolist()
    y_train_list = y_train.values.tolist()
    pred_test_list = result_test["Predict"].values.tolist()
    y_test_list = y_test.values.tolist()
    # 混合行列
    confusion_train = confusion_matrix(y_train_list, pred_train_list)
    confusion_test = confusion_matrix(y_test_list, pred_test_list)
    print("------confusion matrix (train)------")
    print(confusion_train,"\n")
    print("------confusion matrix (test)------")
    print(confusion_test,"\n")


main()
