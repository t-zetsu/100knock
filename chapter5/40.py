input_file = 'ai.ja.txt.parsed'

class Morph:
  def __init__(self, components):
    self.surface = components[0]
    self.base = components[2]
    self.pos = components[3]
    self.pos1 = components[5]

def get_morph(lines):
    output = []
    tmp = []
    for line in lines:
        components = line.split()
        if components[0] == "+" or components[0] == "*" or components[0] == "#":
            continue
        elif components[0] == "EOS":
            output.append(tmp)
            tmp = []
        else:
            tmp.append(Morph(components))
    return output

def main():
    with open(input_file, encoding='utf-16') as f:
        lines = f.readlines()
    output = get_morph(lines)

    for unit in output[1]:
        print(vars(unit))
        
main()

