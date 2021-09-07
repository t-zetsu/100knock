from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd
import pickle

model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True) 
data = pd.read_csv('data/questions-words.txt', sep=' ', names=["word0", "word1","word2","word3"])
output_file = 'outputs/64.csv'

def cos_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def most_similarity(vec, n):
    most = [("",0)] * n
    vocab = model.index_to_key
    for word2 in vocab:
        sim_tmp = cos_similarity(vec,model[word2])
        for i, (key, sim) in enumerate(most):
            if sim_tmp > sim:
                most[i] = (word2, sim_tmp)
                most.sort(key = lambda x: x[1]) 
                break
    return most

def main(): 
    category = "error"
    words = []
    cos = []
    cate = []
    for i, row in data.iterrows():
        if row[0]!=":":
            w, c = model.most_similar(positive=[row[1], row[2]], negative=[row[0]], topn=1)[0]
            words.append([row[0],row[1],row[2],row[3],w])
            cos.append(c)
            cate.append(category)
        else:
            category = row[1]
    df_words = pd.DataFrame(words,columns=["word1","word2","word3","word_ans","word_pred"])
    df_cos = pd.DataFrame(cos,columns=["cos"])
    df_cate = pd.DataFrame(cate, columns=["category"])
    output = pd.concat([df_cate,df_words, df_cos], axis=1)
    output.to_csv(output_file, sep='\t')

main()