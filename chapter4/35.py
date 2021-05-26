# coding: utf-8
parsed_file = "neko.txt.mecab"
output_file = "output/35.txt"

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

# 任意の品詞の連接を抽出（最長一致）
def extract_conjunction(lines,pos):
    conjection = []
    nouns = ""
    n = 1
    for line in lines:
        counter = 1
        for i in range(len(line)-n):
            if line[i]["pos"] == pos:
                if line[i+1]["pos"] == pos:
                    counter += 1
                elif counter == n:
                    for j in range(counter):
                        nouns += line[i+j+1-counter]["surface"]
                    conjection.append(nouns+"\n")
                    nouns = ""
                    counter = 1
                elif counter > n:
                    conjection = []
                    n = counter
                    for j in range(counter):
                        nouns += line[i+j+1-counter]["surface"]
                    conjection.append(nouns+"\n")
                    nouns = ""
                    counter = 1
                elif counter < n:
                    counter = 1
    
    return n, conjection




def main():
    lines = form_sentences(parsed_file)
    n, output = extract_conjunction(lines, "名詞")

    with open(output_file, mode='w', encoding='utf-8') as f:
        f.writelines([f"-----{n}連続名詞-----\n"])
        f.writelines(output)
    
main()