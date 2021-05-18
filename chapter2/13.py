output = []

with open("col1.txt", mode='r', encoding='utf-8') as f:
    col1 = f.readlines()
with open("col2.txt", mode='r', encoding='utf-8') as f:
    col2 = f.readlines()

for i in range(len(col1)):
    output.append(col1[i].replace("\n","") + "\t" + col2[i] + "\n")

with open("col.1-2.txt", mode='w', encoding='utf-8') as f:
    f.writelines(output)
