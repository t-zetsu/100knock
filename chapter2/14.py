import sys
args = sys.argv
N = int(args[1])
output = []

with open("hightemp.txt", mode='r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(N):
    print(lines[i].replace("\n",""))
