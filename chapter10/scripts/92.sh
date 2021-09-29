export PYTHONIOENCODING=utf-8
DATA=data-bin/kyoto
CHECKPOINT=checkpoints/kyoto/checkpoint_best.pt 
INPUT=data/kyoto/test.ja
fairseq-interactive \
    $DATA \
    --path $CHECKPOINT \
    --input $INPUT \
    --task translation \
    | grep '^H' | cut -f3 > outputs/92.txt
