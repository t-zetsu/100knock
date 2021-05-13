def ngram(str,n):
    gram=[]
    for i in range(len(str)-n+1):
        gram.append(str[i:i+n])
    return gram

str1 = "paraparaparadise"
str2 = "paragraph"
X = set(ngram(str1,2))
Y = set(ngram(str2,2))

print(X|Y)
print(X-Y)
print(X&Y)
print("se" in X)
print("se" in Y)
