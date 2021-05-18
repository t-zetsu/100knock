d = {}
with open("hightemp.txt", mode='r', encoding='utf-8') as f:
    for line in f:
        d[line.replace("\n","")] = float(line.split()[2]) 

d_sorted = sorted(d.items(), key=lambda x:x[1], reverse=True)

for i in range(len(d_sorted)):
    print(d_sorted[i][0])