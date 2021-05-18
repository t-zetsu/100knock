import sys
args = sys.argv
N = int(args[1])

with open("hightemp.txt", mode='r', encoding='utf-8') as f:
    lines = f.readlines()

L = int(len(lines)/N)
for i in range(N):
    output = lines[i*L:(i+1)*L]
    with open(f"split{i+1}.txt", mode='w', encoding='utf-8') as f:
        f.writelines(output)

