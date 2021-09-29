import sys, re

# 入出力のパス

args = sys.argv
input_path = args[1]
output_path = args[2]
print("\n******"+input_path[4:].rstrip(".log")+"*******")

# Grep検索対象
grep_target = 'H-'


# ファイル読み込みとGrep
with open(input_path, mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    GREP_TARGET = [line for line in lines if line.startswith(grep_target)]

GREP_TARGET = sorted(GREP＿TARGET, key=lambda s: int(re.search(r'\d+', s).group()))
GREP_TARGET = map(lambda s: re.sub(f'{grep_target}\d+','',s),GREP_TARGET)
GREP_TARGET = map(lambda s: re.sub('-\d+.\d+','',s),GREP_TARGET)
GREP_TARGET = map(lambda s: s.lstrip(),GREP_TARGET)

output = map(lambda s: s.replace("<unk>",""),GREP_TARGET)

# ファイル出力
with open(output_path, mode='w', encoding='utf-8') as f:
    f.writelines(output)