import random

def shuffle(str):
    l = str.split()
    for i in range(len(l)):
        if len(l[i]) > 4:
            l[i] = l[i][0] + "".join(random.sample(l[i][1:-1],len(l[i][1:-1]))) + l[i][-1]
    return " ".join(l)

str = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
print(shuffle(str))
