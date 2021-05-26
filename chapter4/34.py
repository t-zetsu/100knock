# coding: utf-8
parsed_file = "neko.txt.mecab"
output_file = "output/34.txt"

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

# 任意の文字列によって結合している名詞句を抽出
def extract_np(lines,middle_string):
    output = []
    for line in lines:
        for i in range(1,len(line)-1):
            if line[i]["surface"] == middle_string:
                if line[i-1]["pos"]=="名詞" and line[i+1]["pos"] == "名詞":
                    output.append(line[i-1]["surface"]+line[i]["surface"]+line[i+1]["surface"]+"\n")
    return output


def main():
    lines = form_sentences(parsed_file)
    output = extract_np(lines, "の")

    with open(output_file, mode='w', encoding='utf-8') as f:
        f.writelines(output)
    
main()