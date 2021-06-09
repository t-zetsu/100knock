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

def extract_X(chunk):
    X = ""
    flag = 0
    for morph in chunk.morphs:
        if morph.pos == "名詞":
            if flag == 0:
                X += "X"
            flag = 1
        elif morph.pos != "特殊" and morph.pos != "名詞":
            X += morph.surface
    return X

def extract_Y(chunk):
    Y = ""
    flag = 0
    for morph in chunk.morphs:
        if morph.pos == "名詞":
            if flag == 0:
                Y += "Y"
            flag = 1
        elif morph.pos != "特殊":
            Y += morph.surface
    return Y

def extract_XtoY(pair,sentence):
    X_chunk = pair[0]
    Y_chunk = pair[1]
    X = "".join([morph.surface for morph in X_chunk.morphs])
    Y = "".join([morph.surface for morph in Y_chunk.morphs])
    path = []
    dst = X_chunk.dst
    while dst != -1:
        next_chunk = sentence[dst]
        N = "".join([morph.surface for morph in next_chunk.morphs])
        path.append(N)
        if Y == N:
            return path[:-1]
        dst = next_chunk.dst
    return []

def extract_XcrossY(pair, sentence):
    X_chunk = pair[0]
    Y_chunk = pair[1]
    X_path = []
    Y_path = []
    X_dst = X_chunk.dst
    Y_dst = Y_chunk.dst
    while X_dst != -1:
        X_path.append("".join([morph.surface if morph.pos != "特殊" else "" for morph in X_chunk.morphs]))
        X_chunk = sentence[X_dst]
        X_dst = X_chunk.dst
    while Y_dst != -1:
        Y_path.append("".join([morph.surface if morph.pos != "特殊" else "" for morph in Y_chunk.morphs]))
        Y_chunk = sentence[Y_dst]
        Y_dst = Y_chunk.dst
    for x in X_path:
        for y in Y_path:
            if x == y:
                return x
    return ""


def extract_path(chunk, sentence, K):
    path = []
    next_chunk = ""
    dst = chunk.dst
    while dst != -1:
        next_chunk = sentence[dst]
        N = "".join([morph.surface if morph.pos != "特殊" else "" for morph in next_chunk.morphs])
        if N == K:
            return path
        path.append(N)
        dst = next_chunk.dst
    return ["ERROR"]
        

def select_pairs(sentence):
    pairs = []
    for chunk_i in sentence:
        idx = sentence.index(chunk_i)
        if check_noun(chunk_i):
            X = chunk_i
            for chunk_j in sentence[idx+1:]:
                if check_noun(chunk_j):
                    Y = chunk_j
                    pairs.append([X,Y])
    return pairs



def extract_noun_pair_path(sentences):
    pair_path = ""
    pair_paths = []
    for sentence in sentences:
        pairs = select_pairs(sentence)
        for pair in pairs:
            X = "".join(extract_X(pair[0]))
            Y = "".join(extract_Y(pair[1]))
            XtoY = extract_XtoY(pair,sentence)
            K = extract_XcrossY(pair,sentence)
            if XtoY != []:
                pair_path = X + "  →  " + "  →  ".join(XtoY) + "  →  " + Y
                pair_paths.append(pair_path)
            elif K != "":
                X_path = extract_path(pair[0],sentence,K)
                Y_path = extract_path(pair[1],sentence,K)
                pair_path = X + "  →  "+  "  →  ".join(X_path) + " | " + Y + "  →  " + "  →  ".join(Y_path) + " | " + K
                pair_paths.append(pair_path)
    return pair_paths




def main():
    with open(sentence_file, encoding='utf-16') as f:
        lines = f.readlines()
    sentences = get_chunk(lines)
    pair_paths = extract_noun_pair_path(sentences)

    for pair_path in pair_paths:
        print(pair_path)
    


main()