mkdir outputs/94
DATA=data-bin/kyoto
CHECKPOINT=checkpoints/kyoto/checkpoint_best.pt 
INPUT=data/kyoto/test.ja
for N in `seq 1 20` ; do
    fairseq-interactive \
    $DATA \
    --path $CHECKPOINT \
    --input $INPUT \
    --beam $N \
    --task translation \
    | grep '^H' | cut -f3 > outputs/94/$N.out.txt
done

for N in `seq 1 20` ; do
    fairseq-score --sys outputs/94/$N.out.txt --ref data/kyoto/test.en > outputs/94/$N.score.txt
done