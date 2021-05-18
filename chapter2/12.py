col1 = [] 
col2 = []

with open("hightemp.txt", mode='r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(len(lines)):
    line = lines[i].split()
    col1.append(line[0]+"\n")
    col2.append(line[1]+"\n")

with open("col1.txt", mode='w', encoding='utf-8') as f:
    f.writelines(col1)
with open("col2.txt", mode='w', encoding='utf-8') as f:
    f.writelines(col2)