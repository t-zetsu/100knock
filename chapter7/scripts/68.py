import pandas as pd
import numpy as np 
from gensim.models.keyedvectors import KeyedVectors
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

input_file = "data/countries-list.csv"
output_file = "outputs/68.png"
model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True) 

def main():
    data = pd.read_csv(input_file, usecols=[1]).sample(n=20)

    countries = data.iloc[:,0].values.tolist()
    countries_vec = [model[country] for country in countries]

    plt.figure(figsize=(20, 5))
    Z = linkage(countries_vec, method='ward')
    dendrogram(Z, labels=countries)
    plt.savefig(output_file)
            
main()