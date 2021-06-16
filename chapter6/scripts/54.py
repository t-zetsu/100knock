from sklearn.linear_model import LogisticRegression
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

# 正解率を計算
def cal_correct(pred, ref):
    correct = 0
    for p, r in zip(pred, ref):
        if p == r:
            correct += 1
    return correct/len(pred)

def main():
    category = {0:"b", 1:"t", 2:"e", 3:"m"}
    # 予測
    result_train = predict(model, x_train)
    result_test = predict(model, x_test)
    result_train.replace(category, inplace=True)
    result_test.replace(category, inplace=True)

    # 正解率を計算
    correct_train = cal_correct(result_train["Predict"],y_train)
    correct_test = cal_correct(result_test["Predict"],y_test)

    print(correct_train)
    print(correct_test)

main()


