str = "I am an NLPer"
def ngram(str,n):
    gram=[]
    for i in range(len(str)-n+1):
        gram.append(str[i:i+n])
    return gram

print(ngram(str.split(),2))
print(ngram(str,2))