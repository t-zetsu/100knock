import re
input_file = "england.txt"

def startswith_key(line, key):
    flag = 0
    for k in key:
        if line.startswith(k):
            flag = 1
    return flag

def main():
    key = "="
    
    with open(input_file, mode='r', encoding='utf-8') as f:
        for line in f:
            if startswith_key(line, key) == 1:
                start = re.search(fr"{key}+",line).end()
                end = -start - 1
                print(line[start:end].replace(" ","") + f":{start}" )

main()