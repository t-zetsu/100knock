import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv('/work/chapter5/data/train.txt', sep='\t', header=None)
val = pd.read_csv('/work/chapter5/data/valid.txt', sep='\t', header=None)
test = pd.read_csv('/work/chapter5/data/test.txt', sep='\t', header=None)

train_val = pd.concat([train, val], axis=0)

vectorizer = CountVectorizer(min_df=3)
train_val_bow = vectorizer.fit_transform(train_val[1])
test_bow =vectorizer.transform(test[1])

train_val_bow_ = pd.DataFrame(train_val_bow.toarray(), columns=vectorizer.get_feature_names())

x_train = train_val_bow_[:len(train)]
x_valid = train_val_bow_[len(train):]
x_test = pd.DataFrame(test_bow.toarray(), columns=vectorizer.get_feature_names())


x_train.to_csv('/work/chapter5/data/train.feature.txt', sep='\t',header=None, index=False)
x_valid.to_csv('/work/chapter5/data/valid.feature.txt', sep='\t',header=None, index=False)
x_test.to_csv('/work/chapter5/data/test.feature.txt', sep='\t',header=None, index=False)
