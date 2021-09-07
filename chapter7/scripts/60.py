from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True) 
vec = model["United_States"]

print(vec)
