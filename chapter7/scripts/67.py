import pandas as pd
import numpy as np 
from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import KMeans

input_file = "outputs/64.csv"
model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True) 

def extract_countries(x):
    if x[0] in ['capital-common-countries', 'capital-world']:
        return x[2]
    elif x[0] in ['currency', 'gram6-nationality-adjective']:
        return x[1]
    
def shape_countries(data):
    countries = data.apply(extract_countries,axis=1).values.tolist()
    countries = list(set(countries))
    countries = [s for s in countries if s != None]
    return countries


def main():
    data = pd.read_csv(input_file, sep='\t')
    data = data.drop(columns=data.columns[0])

    countries = shape_countries(data)
    countries_vec = [model[country] for country in countries]

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(countries_vec)
    for i in range(5):
        cluster = np.where(kmeans.labels_ == i)[0]
        print('cluster', i)
        print(', '.join([countries[k] for k in cluster]))
            
main()
