#!/bin/bash

LC_ALL=C.UTF-8
LANG=C.UTF-8
DATA=kyoto/data-bin
MODEL=kyoto/model.pt


echo -e $1 | python3 tok.py \
| fairseq-interactive 2> error.log\
    $DATA \
    --path $MODEL \
    --beam 10 \
    --task translation \
    | grep '^H' | cut -f3 
