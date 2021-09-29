#!/usr/bin/env python3

"""データを作る"""

import argparse
import sys
import io
import numpy as np
from sklearn.model_selection import train_test_split

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

def main(args):
    src = args.src
    tgt = args.tgt
    output = args.output
    source = []
    target = []
    for line in sys.stdin:
        source.append(line.split("\t")[3].rstrip())
        target.append(line.split("\t")[2])
    print(source[0])
    print(target[0])
    source = np.array(source)
    target = np.array(target)
    source_train, source_tmp, target_train, target_tmp = train_test_split(source, target, test_size=0.4)
    source_valid, source_test, target_valid, target_test = train_test_split(source_tmp, target_tmp, test_size=0.5)
    with open(output+'/train.'+src, 'w', encoding='utf-8') as f:
        for line in source_train:
            print(line, file=f)
    with open(output+'/train.'+tgt, 'w', encoding='utf-8') as f:
        for line in target_train:
            print(line, file=f)
    with open(output+'/valid.'+src, 'w', encoding='utf-8') as f:
        for line in source_valid:
            print(line, file=f)
    with open(output+'/valid.'+tgt, 'w', encoding='utf-8') as f:
        for line in target_valid:
            print(line, file=f)
    with open(output+'/test.'+src, 'w', encoding='utf-8') as f:
        for line in source_test:
            print(line, file=f)
    with open(output+'/test.'+tgt, 'w', encoding='utf-8') as f:
        for line in target_test:
            print(line, file=f)

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", "-s", type=str, default="ja", help="source lang")
    parser.add_argument("--tgt", "-t", type=str, default="en", help="target lang")
    parser.add_argument("--output", "-o", type=str, default="output", help="output dir")
    args = parser.parse_args()

    main(args)