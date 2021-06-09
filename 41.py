input_file = 'ai.ja.txt.parsed'

import re

class Morph:
    def __init__(self, components):
        self.surface = components[0]
        self.base = components[2]
        self.pos = components[3]
        self.pos1 = components[5]

class Chunk:
    def __init__(self, dst):
        self.morphs = []
        self.dst = dst
        self.srcs = []

def get_chunk(lines):
    output = []
    sent = []
    chunk = None

    for line in lines:
        components = line.split()
        # 文が*で始まるとき，chunkオブジェクトを作成
        if components[0] == "*": 
            dst = int(re.sub("[A-Z]","",components[1]))
            chunk = Chunk(dst)
            sent.append(chunk)

        # 文が+ or #で始まるとき，何もしない
        elif components[0] == "+" or components[0] == "#":
            continue

        # 文がEOSで始まるとき，各分節の係り受け元を求める
        elif components[0] == "EOS":
            for i, c in enumerate(sent):
                if c.dst == -1:
                    continue
                else:
                    sent[c.dst].srcs.append(i)
            output.append(sent)
            sent = []

        # 文が単語で始まるとき，morphオブジェクトを作成
        else:
            morph = Morph(components)
            sent[-1].morphs.append(morph)

    return output


def main():
    with open(input_file, encoding='utf-16') as f:
        lines = f.readlines()
    output = get_chunk(lines)

    for idx, unit in enumerate(output[1]):
        print(idx,[m.surface for m in unit.morphs],"係り先:",unit.dst)
        
main()