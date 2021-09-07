from gensim.models.keyedvectors import KeyedVectors
import numpy as np

def cos_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True) 

print(cos_similarity(model["United_States"],model["U.S."]))

