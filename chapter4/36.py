# coding: utf-8
from collections import Counter
parsed_file = "neko.txt.mecab"
output_file = "output/36.txt"

# 形態素解析済みのテキストを成形
def form_sentences(parsed_file):
    morphemes = []
    with open(parsed_file, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.replace("\n","")
            if line != "EOS":
                cols = line.split("\t")
                cols_split = cols[1].split(",")
                morpheme = {
                    "surface":cols[0],
                    "base":cols_split[6],
                    "pos":cols_split[0],
                    "pos1":cols_split[1]
                }
                morphemes.append(morpheme)
            else:
                if morphemes != []:
                    yield morphemes
                morphemes = []

# 単語の出現頻度をカウント
def word_counter(lines):
    counter = Counter()
    for line in lines:
        counter.update([word["surface"] for word in line])
    return counter.most_common()


def main():
    output = []
    lines = form_sentences(parsed_file)
    counted_list = word_counter(lines)
    for c in counted_list:
        output.append(f"{c}\n")
    with open(output_file, mode='w', encoding='utf-8') as f:
        f.writelines(output)
    
main()