def chipher(str):
    l = []
    for i in range(len(str)):
        if str[i].islower():
            l.append(219-ord(str[i]))
        else:
            l.append(str[i])
    return l

def decode(l):
    str = ""
    for i in range(len(l)):
        if isinstance(l[i],int):
            str += chr(219-l[i])
        else:
            str += l[i]
    return str

print(chipher("This"))
print(decode(chipher("This")))