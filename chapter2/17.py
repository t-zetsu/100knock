pref = []
with open("hightemp.txt", mode='r', encoding='utf-8') as f:
    for line in f:
        pref.append(line.split()[0])

print(set(pref))