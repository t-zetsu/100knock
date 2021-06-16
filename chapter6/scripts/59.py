import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# データのロード
x_train = pd.read_csv("/work/chapter6/data/train.feature.txt", sep='\t', header=None)
x_valid = pd.read_csv("/work/chapter6/data/valid.feature.txt", sep='\t', header=None)
x_test = pd.read_csv('/work/chapter6/data/test.feature.txt', sep='\t', header=None)
y_train = pd.read_csv('/work/chapter6/data/train.txt', sep='\t', header=None)[0]
y_valid = pd.read_csv('/work/chapter6/data/valid.txt', sep='\t', header=None)[0]
y_test = pd.read_csv('/work/chapter6/data/test.txt', sep='\t', header=None)[0]

def cal_correct(pred, ref):
    correct = 0
    for p, r in zip(pred, ref):
        if p == r:
            correct += 1
    return correct/len(pred)


def calc_scores(C,solver):
    #モデルの宣言
    model = LogisticRegression(random_state=0, max_iter=10000, C=C, solver=solver)
    #モデルの学習
    model.fit(x_train, y_train)
    #モデルの検証
    y_train_pred = model.predict(x_train)
    y_valid_pred = model.predict(x_valid)
    y_test_pred = model.predict(x_test)
    #スコア
    scores = []
    scores.append(cal_correct(y_train_pred,y_train))
    scores.append(cal_correct(y_valid_pred,y_valid))
    scores.append(cal_correct(y_test_pred,y_test))
    return scores

def main():
    # Cとsolverの総当たり
    C = np.logspace(-5, 1, 5, base=10)
    solver = ["lbfgs","sag"]
    best_parameter = None
    best_scores = None
    max_valid_score = 0
    #itertools.product()で全ての組み合わせを作成
    for c, s in itertools.product(C, solver):
        # ハイパーパラメータの組み合わせの表示
        print(c, s)
        #ハイパーパラメータの組み合わせで関数の実行
        scores = calc_scores(c, s)
        #前のスコアより高ければ結果を更新
        if scores[1] > max_valid_score:
            max_valid_score = scores[1]
            best_parameter = [c, s]
            best_scores = scores
    #最適なハイパーパラメータの組み合わせとスコアの表示
    print ('C: ', best_parameter[0], 'solver: ', best_parameter[1])
    print ('best scores: ', best_scores)
    print ('test accuracy: ', best_scores[2])

main()