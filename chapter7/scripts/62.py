from gensim.models.keyedvectors import KeyedVectors
import numpy as np

model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True) 

def cos_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def most_similarity(word1, n):
    most = [("",0)] * n
    vocab = model.index_to_key
    for word2 in vocab:
        sim_tmp = cos_similarity(model[word1],model[word2])
        for i, (key, sim) in enumerate(most):
            if sim_tmp > sim and word1!=word2:
                most[i] = (word2, sim_tmp)
                most.sort(key = lambda x: x[1]) 
                break
    return most

def main():  
    most_sim = most_similarity("United_States", 10)
    most_sim.reverse()
    for m in most_sim:
        print(m)

main()

