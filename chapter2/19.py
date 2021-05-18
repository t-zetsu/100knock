import collections
pref = []
with open("hightemp.txt", mode='r', encoding='utf-8') as f:
    for line in f:
        pref.append(line.split()[0])

c = collections.Counter(pref)
print(c.most_common())