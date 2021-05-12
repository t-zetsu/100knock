str = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.".split()

n = []
for i in range(len(str)):
    n.append(len(str[i].strip(",.")))

print(n)