sentence_file = 'ai.ja.txt.parsed'

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
    sentences = []
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
            sentences.append(sent)
            sent = []

        # 文が単語で始まるとき，morphオブジェクトを作成
        else:
            morph = Morph(components)
            sent[-1].morphs.append(morph)

    return sentences

# 名詞チェック
def check_noun(chunk):
    for morph in chunk.morphs:
        if morph.pos == "名詞":
            return True
        return False

def search_path(chunk,sentence,path):
    dst = chunk.dst
    tmp = [morph.surface for morph in chunk.morphs]
    path.append("".join(tmp))
    if dst == -1:
        return "  →  ".join(path)
    chunk = sentence[dst]
    return search_path(chunk,sentence,path)


        
# 名詞から根へのパス抽出
def extract_path(sentences):
    path = ""
    paths = []
    for sentence in sentences:
        for chunk in sentence:
            if check_noun(chunk) and chunk.dst != -1:
                path = search_path(chunk,sentence,[])
                paths.append(path)
    return paths




def main():
    with open(sentence_file, encoding='utf-16') as f:
        lines = f.readlines()
    sentences = get_chunk(lines)
    paths = extract_path(sentences)

    for path in paths:
        print(path)
    


main()