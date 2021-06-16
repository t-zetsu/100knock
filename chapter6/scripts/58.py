from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np

# データのロード
x_train = pd.read_csv("/work/chapter6/data/train.feature.txt", sep='\t', header=None)
x_valid = pd.read_csv("/work/chapter6/data/valid.feature.txt", sep='\t', header=None)
x_test = pd.read_csv('/work/chapter6/data/test.feature.txt', sep='\t', header=None)
y_train = pd.read_csv('/work/chapter6/data/train.txt', sep='\t', header=None)[0]
y_valid = pd.read_csv('/work/chapter6/data/valid.txt', sep='\t', header=None)[0]
y_test = pd.read_csv('/work/chapter6/data/test.txt', sep='\t', header=None)[0]

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
    result = []
    C = np.logspace(-5, 4, 10, base=10)
    for c in C:
        #モデルの学習
        model = LogisticRegression(random_state=0, max_iter=5000, C=c)
        model.fit(x_train, y_train)
        #それぞれの予測値
        train_predict = predict(model, x_train)
        valid_predict = predict(model, x_valid)
        test_predict = predict(model, x_test)
        #正解率の計算
        train_score = cal_correct(train_predict["Predict"],y_train)
        valid_score = cal_correct(valid_predict["Predict"],y_valid)
        test_score = cal_correct(test_predict["Predict"],y_test)
        #resultに格納
        result.append([c, train_score, valid_score, test_score])
    result = np.array(result).T
    
    #可視化
    fig = plt.figure()
    plt.plot(result[0], result[1], label='train')
    plt.plot(result[0], result[2], label='valid')
    plt.plot(result[0], result[3], label='test')
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.xlabel('C')
    plt.legend()
    fig.savefig("output/58.png")



main()