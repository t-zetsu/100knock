input_path = "ai.ja.txt"
output_path = "ai.ja.txt.edited"

with open(input_path, encoding="UTF-8") as f:
    lines = f.readlines()

lines = [line for line in lines if line != "\n"]
lines = [line.replace(" ","") for line in lines]
output = []
for line in lines:
    line_splits = line.replace("\n","").split("。")
    for line_split in line_splits:
        if line_split != "":
            output.append(line_split + "。" + "\n")
lines = output
output = []
for i in range(len(lines)):
    line = lines[i]
    if i < len(lines)-1:
        line_next = lines[i+1]
    if line_next[0] == "」" or line_next[0] == ")":
        output.append(line.replace("\n",""+line_next))
    elif line[0] != "」" and line[0] != ")":
        output.append(line)



with open(output_path, mode="w", encoding="UTF-8") as f:
    f.writelines(output)