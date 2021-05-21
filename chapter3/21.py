input_file = "england.txt"

def find_key(line, key):
    flag = 0
    for k in key:
        if k in line:
            flag = 1
    return flag

def main():
    key = ["Category"]
    
    with open(input_file, mode='r', encoding='utf-8') as f:
        for line in f:
            if find_key(line, key) == 1:
                print(line[:-1])

main()