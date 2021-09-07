import pandas as pd
from gensim.models.keyedvectors import KeyedVectors

input_file = "data/combined.csv"
model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True) 

def spearman(df):
    D = df.apply(lambda x: (x[2]-x[3])**2,axis = 1)
    N = len(df)
    r = 1 - (6*D[0].sum())/(N**3-N)
    return r

def main():
    data = pd.read_csv(input_file)
    data['sim']  = data.apply(lambda x: model.similarity(x[0],x[1]),axis = 1)
    r = spearman(data.rank())

    print(f'スピアマン相関係数: {r}')
            
main()