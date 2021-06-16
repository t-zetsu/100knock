from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
import numpy as np

# モデルのロード
filename = "/work/chapter5/model/model.sav"
model = pickle.load(open(filename, 'rb'))

# データのロード
x_test = pd.read_csv('/work/chapter5/data/test.feature.txt', sep='\t', header=None)

# 予測結果とその確率を表示
def predict(model, data):
    prob = pd.DataFrame(np.max(model.predict_proba(data), axis=1))
    prob.columns = ["Probability"]
    pred = pd.DataFrame(model.predict(data))
    pred.columns = ["Predict"]
    result = pd.concat([prob, pred], axis = 1)
    return result

# x_testを予測
result_test = predict(model, x_test)
category = {0:"b", 1:"t", 2:"e", 3:"m"}
result_test.replace(category, inplace=True)

# 結果を保存
result_test.to_csv('/work/chapter5/output/53.txt', sep='\t',header=False, index=False)
