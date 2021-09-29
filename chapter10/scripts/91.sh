# ===========
# preprocess
# ===========
# DATA_DIR=data/kyoto
# DEST_DIR=data-bin/kyoto
# fairseq-preprocess -s ja -t en \
#     --trainpref $DATA_DIR/train --validpref $DATA_DIR/valid --testpref $DATA_DIR/test \
#     --tokenizer moses --bpe subword_nmt \
#     --destdir $DEST_DIR \
#     --workers 20 


# ===========
# training
# ===========
# =================実行コマンド===================
# LOG=outputs/91.txt ; nohup bash 91.sh >> $LOG 2> log/error.log &
# ==============================================
DATA=data-bin/kyoto
CHECKPOINT=checkpoints/kyoto
TENSORBOARD=tensorboard/kyoto
CUDA_VISIBLE_DEVICES=2 fairseq-train \
    $DATA\
    --save-dir $CHECKPOINT/ \
    --tensorboard-logdir $TENSORBOARD \
    --arch transformer --task translation \
	--share-decoder-input-output-embed \
 	--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
	--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--dropout 0.3 --weight-decay 0.0001 \
	--max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
	--fp16 \
    --no-epoch-checkpoints