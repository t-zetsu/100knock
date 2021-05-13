str = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.".strip(".").split()
n = [1, 5, 6, 7, 8, 9, 15, 16, 19]
d={}
for i in range(len(str)):
    if i+1 in n:
        d[str[i][0]] = i+1
    else:
        d[str[i][0:2]] = i+1
print(d)
