import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/work/chapter5/data/newsCorpora.csv', sep='\t', header=None)
data.columns = ['ID','TITLE','URL','PUBLISHER','CATEGORY','STORY','HOSTNAME','TIMESTAMP']
df = data[data['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])]
df = df.sample(frac=1)

train, tmp = train_test_split(df, test_size=0.2)
valid, test = train_test_split(tmp, test_size=0.5)

train.to_csv('/work/chapter5/data/train.txt', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
valid.to_csv('/work/chapter5/data/valid.txt', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
test.to_csv('/work/chapter5/data/test.txt', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)