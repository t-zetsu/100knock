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

# 動詞チェック
def check_verb(chunk):
    for morph in chunk.morphs:
        if morph.pos == "動詞":
            return True
        return False

# 動詞の基本形を取り出す
def extract_verb_base(chunk):
    for morph in chunk.morphs:
        if morph.pos == "動詞":
            return morph.base

# 助詞を取り出す
def extract_pp_particle(chunk):
    for morph in chunk.morphs:
        if morph.pos == "助詞":
            return morph.base
        

# 格パターン抽出
def extract_case_pattern(sentences):
    case_patterns = []
    for sentence in sentences:
        for chunk in sentence:
            pp_particle = []
            src_chunk = []
            if chunk.srcs != [] and check_verb(chunk):
                for idx in chunk.srcs:
                    pp_particle.append(extract_pp_particle(sentence[idx]))
                pp_particle = sorted([p for p in pp_particle if p != None])
                case_pattern = extract_verb_base(chunk) + "\t" + " ".join(pp_particle)
                case_patterns.append(case_pattern)
    return case_patterns


def main():
    with open(sentence_file, encoding='utf-16') as f:
        lines = f.readlines()
    sentences = get_chunk(lines)
    case_patterns = extract_case_pattern(sentences)

    for case_pattern in case_patterns:
        print(case_pattern)
    



main()