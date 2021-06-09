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

# サ変接続名詞+を チェック
def check_sahen(chunk):
    for morph in chunk.morphs:
        idx = chunk.morphs.index(morph)
        if idx+1 < len(chunk.morphs):
            if morph.pos1 == 'サ変名詞' and chunk.morphs[idx+1].surface == "を":
                return True
    return False

# 動詞の基本形を取り出す
def extract_verb_base(chunk):
    verb_base = ""
    for morph in reversed(chunk.morphs):
        if morph.pos == "動詞":
            verb_base = morph.base
    return verb_base

# サ変接続名詞+を　を取り出す
def extract_sahen(chunk):
    sahen = ""
    for morph in chunk.morphs:
        idx = chunk.morphs.index(morph)
        if idx+1 < len(chunk.morphs):
            if morph.pos1 == 'サ変名詞' and chunk.morphs[idx+1].surface == "を":
                sahen = morph.surface + chunk.morphs[idx+1].surface
    return sahen


# 助詞と結合した文節を取り出す
def extract_chunk(chunk):
    src_chunk = []
    pp_particle = ""
    for morph in chunk.morphs:
        src_chunk.append(morph.surface)
        if morph.pos == "助詞":
            pp_particle = morph.base
    if pp_particle != "":
        return pp_particle, "".join(src_chunk)
    else:
        return "", ""

        
# 格パターン&格フレームの抽出
def extract_case(sentences):
    case_patterns = []
    for sentence in sentences:
        for chunk in sentence:
            pp_particle = []
            src_chunk = []
            predicate = ""
            if chunk.srcs != [] and check_verb(chunk):
                for idx in chunk.srcs:
                    tmp_pp, tmp_src = extract_chunk(sentence[idx])
                    pp_particle.append(tmp_pp)
                    src_chunk.append(tmp_src)                    
                    if check_sahen(sentence[idx]):
                        sahen = extract_sahen(sentence[idx])
                        sahen_chunk = tmp_src
                        verb_base = extract_verb_base(chunk)
                        predicate = sahen + verb_base 
                if predicate != "": 
                    pp_particle.remove("を")
                    src_chunk.remove(sahen_chunk)  
                    try:     
                        tmp = zip(pp_particle,src_chunk)
                        tmp = sorted(tmp, key=lambda x: x[0])
                        pp_particle,src_chunk = zip(*tmp)
                    except:
                        continue
                    case_pattern = predicate + "\t" + " ".join(pp_particle) + "\t" + " ".join(src_chunk)
                    case_patterns.append(case_pattern)
    return case_patterns


def main():
    with open(sentence_file, encoding='utf-16') as f:
        lines = f.readlines()
    sentences = get_chunk(lines)
    case_patterns = extract_case(sentences)

    for case_pattern in case_patterns:
        print(case_pattern)
    



main()