#!/usr/bin/env python3

"""サブワード分割"""

import argparse
import sentencepiece as spm
import re


def main(args):
    lang = args.lang
    input_path = args.input
    output_path = args.output
    if lang == "ja":
        spm.SentencePieceTrainer.Train(f"--input={input_path}/train.ja   --model_prefix=jpara --vocab_size=16000 --character_coverage=0.9995")
        sp = spm.SentencePieceProcessor()
        sp.Load("jpara.model")
        for src, dst in [
            (f"{input_path}/train.ja", f"{output_path}/train.ja"),
            (f"{input_path}/valid.ja", f"{output_path}/valid.ja"),
            (f"{input_path}/test.ja", f"{output_path}/test.ja"),
        ]:
            with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
                for x in f:
                    x = x.strip()
                    x = re.sub(r"\s+"," ",x)
                    x = sp.encode_as_pieces(x)
                    x = " ".join(x)
                    print(x, file=g)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", "-l", type=str, default="ja", help="lang")
    parser.add_argument("--input", "-i", type=str, default="data", help="input file")
    parser.add_argument("--output", "-o", type=str, default="data", help="output file")
    args = parser.parse_args()

    main(args)