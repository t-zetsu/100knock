input_file = 'ai.ja.txt.parsed'
output_file = "output/44.png"

import re
import pydot_ng

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
    outputs = []
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
            outputs.append(sent)
            sent = []

        # 文が単語で始まるとき，morphオブジェクトを作成
        else:
            morph = Morph(components)
            sent[-1].morphs.append(morph)

    return outputs

# 文節に名詞が含まれる場合1をかえす
def check_noun(morphs):
    flag = 0
    for m in morphs:
        if m.pos == "名詞":
            flag = 1
    return flag

# 文節に動詞が含まれる場合1をかえす
def check_verb(morphs):
    flag = 0
    for m in morphs:
        if m.pos == "動詞":
            flag = 1
    return flag



def main():
    with open(input_file, encoding='utf-16') as f:
        lines = f.readlines()
    outputs = get_chunk(lines)

    img = pydot_ng.Dot(graph_type="digraph")
    img.set_node_defaults(fontname="Meiryo UI", fontsize="10")

    output = outputs[2]
    for chunk in output:
        if chunk.dst != -1: # 係り受け先がある
            src = "".join([m.surface if m.pos != "特殊" else "" for m in chunk.morphs])
            dst = "".join([m.surface if m.pos != "特殊" else "" for m in output[int(chunk.dst)].morphs])
            img.add_edge(pydot_ng.Edge(src, dst))

    img.write_png(output_file)


main()