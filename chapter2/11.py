text = "hightemp.txt"

with open(text, mode='r', encoding='utf-8') as f:
    lines = f.read()
    print(lines.replace("\t"," "))

