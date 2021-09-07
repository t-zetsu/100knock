import pandas as pd
import numpy as np 
from gensim.models.keyedvectors import KeyedVectors
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

input_file = "data/countries-list.csv"
output_file = "outputs/69.png"
model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True) 

def main():
    data = pd.read_csv(input_file, usecols=[1])

    countries = data.iloc[:,0].values.tolist()
    countries_vec = [model[country] for country in countries]
    
    tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
    countries_emb = tsne.fit_transform(countries_vec)
    
    plt.figure(figsize=(20, 20))
    plt.scatter(np.array(countries_emb).T[0], np.array(countries_emb).T[1])
    for (x, y), name in zip(countries_emb, countries):
        plt.annotate(name, (x, y))
    plt.savefig(output_file)
            
main()