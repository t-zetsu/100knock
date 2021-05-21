input_file = "england.txt"

def find_key(line, key):
    flag = 0
    for k in key:
        if k in line:
            flag = 1
    return flag

def get_idx(line, list_s):
    for s in list_s:
        if s in line:
            idx = line.find(s)+len(s)
    return idx

def main():
    key = ["Category"]
    
    with open(input_file, mode='r', encoding='utf-8') as f:
        for line in f:
            if find_key(line, key) == 1:
                start = get_idx(line,[":"])
                end = get_idx(line,["]"]) - 1
                print(line[start:end])

main()