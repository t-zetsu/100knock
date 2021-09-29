N=3

CHECKPOINT=checkpoints/kyoto/$N
TENSORBOARD=tensorboard/kyoto/$N
TRAIN=log/$N.train
GENERATE=log/$N.generate
DATA=data-bin/kyoto
OUTPUTS=outputs/97.txt


#####generate#####
CUDA_VISIBLE_DEVICES=2 fairseq-generate > $GENERATE 2> $GENERATE.error \
	$DATA \
	--gen-subset test \
	--task translation \
	--path $CHECKPOINT/checkpoint_best.pt \
    --beam 10 --remove-bpe \
	--batch-size 512

echo "========experiment $N========" >> $OUTPUTS
tail -n 1 $GENERATE >> $OUTPUTS


#####train#####
# ==============================================
# experiment 0  91での設定 
# dropout:0.3 lr:5e-4
# ==============================================
# CUDA_VISIBLE_DEVICES=2 fairseq-train > $TRAIN 2> $TRAIN.error\
#     $DATA\
#     --save-dir $CHECKPOINT/ \
#     --tensorboard-logdir $TENSORBOARD \
#     --arch transformer --task translation \
# 	--share-decoder-input-output-embed \
#  	--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
# 	--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
# 	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
# 	--dropout 0.3 --weight-decay 0.0001 \
# 	--max-tokens 4096 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
# 	--fp16 \
#     --no-epoch-checkpoints \
#     --max-epoch 10


# ==============================================
# experiment 1
# dropout:0.1  lr:5e-4
# ==============================================
# CUDA_VISIBLE_DEVICES=2 fairseq-train \
#     $DATA\
#     --save-dir $CHECKPOINT/ \
#     --tensorboard-logdir $TENSORBOARD \
#     --arch transformer --task translation \
# 	--share-decoder-input-output-embed \
#  	--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
# 	--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
# 	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
# 	--dropout 0.1 --weight-decay 0.0001 \
# 	--max-tokens 2048 \
#     --max-epoch 80 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
# 	--fp16 \
#     --no-epoch-checkpoints

# ==============================================
# experiment 2
# dropout:0.3  lr:1e-3
# ==============================================
# CUDA_VISIBLE_DEVICES=2 fairseq-train > $TRAIN 2> $TRAIN.error\
#     $DATA\
#     --save-dir $CHECKPOINT/ \
#     --tensorboard-logdir $TENSORBOARD \
#     --arch transformer --task translation \
# 	--share-decoder-input-output-embed \
#  	--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
# 	--lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
# 	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
# 	--dropout 0.3 --weight-decay 0.0001 \
# 	--max-tokens 4096 \
#     --max-epoch 30 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 10, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
# 	--fp16 \
#     --no-epoch-checkpoints


# ==============================================
# experiment 3
# dropout:0.5  lr:5e-4
# ==============================================
# CUDA_VISIBLE_DEVICES=2 fairseq-train > $TRAIN 2> $TRAIN.error\
#     $DATA\
#     --save-dir $CHECKPOINT/ \
#     --tensorboard-logdir $TENSORBOARD \
#     --arch transformer --task translation \
# 	--share-decoder-input-output-embed \
#  	--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
# 	--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
# 	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
# 	--dropout 0.5 --weight-decay 0.0001 \
# 	--max-tokens 4096 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 10, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
# 	--fp16 \
#     --no-epoch-checkpoints \
#     --max-epoch 20