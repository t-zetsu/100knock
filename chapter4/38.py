# coding: utf-8
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
parsed_file = "neko.txt.mecab"
output_file = "output/38.png"
fp = FontProperties(fname=r'C:/Windows/Fonts/BIZ-UDGothicB.ttc', size=14)

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
    count = []
    lines = form_sentences(parsed_file)
    counted_list = word_counter(lines)
    for c in counted_list:
        count.append(int(c[1]))
    left = np.array(count)
    plt.hist(left,bins=20,range=(1,20))
    plt.xlim(left=1, right=20)
    plt.xlabel('出現頻度', fontproperties=fp)
    plt.ylabel('単語の種類数', fontproperties=fp)
    plt.savefig(output_file)


    
main()