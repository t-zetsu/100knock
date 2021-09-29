task=kyoto.fine
CHECKPOINT=checkpoints/$task
TENSORBOARD=tensorboard/$task
TRAIN=log/$task.train
GENERATE=log/$task.generate
DATA=data-bin/kyoto2
OUTPUTS=outputs/98.txt

# ==============================================
# scripts
# ==============================================
#####スクリプトの実行コマンド
#####japaraデータを整形
# python3 scripts/mkdata.py --src ja --tgt en --output data/j-para < data/jpara/en-ja.bicleaner05.txt
#####日本語データをサブワード化
# python3 scripts/bpe.py -l ja -i data/j-para -o data/j-para.sub
#####英語データをサブワード化
# subword-nmt learn-bpe -s 16000 < data/j-para/train.en > data/j-para.en.codes
# subword-nmt apply-bpe -c data/j-para.en.codes < data/j-para/train.en > data/j-para.sub/train.en
# subword-nmt apply-bpe -c data/j-para.en.codes < data/j-para/valid.en > data/j-para.sub/valid.en
# subword-nmt apply-bpe -c data/j-para.en.codes < data/j-para/test.en > data/j-para.sub/test.en

# ==============================================
# preprocess
# ==============================================
# DATA_DIR=data/j-para.sub
# fairseq-preprocess -s ja -t en \
#     --trainpref $DATA_DIR/train --validpref $DATA_DIR/valid --testpref $DATA_DIR/test \
#     --destdir $DATA \
#     --workers 20 

# DATA_DIR=data/kyoto
# fairseq-preprocess -s ja -t en \
#     --trainpref $DATA_DIR/train --validpref $DATA_DIR/valid --testpref $DATA_DIR/test \
#     --tgtdict data-bin/j-para/dict.en.txt \
#     --srcdict data-bin/j-para/dict.ja.txt \
#     --destdir $DATA \
#     --workers 20 

# ==============================================
# generate
# ==============================================
CUDA_VISIBLE_DEVICES=3 fairseq-generate > $GENERATE 2> $GENERATE.error \
	$DATA \
	--gen-subset test \
	--task translation \
	--path $CHECKPOINT/checkpoint_last.pt \
    --beam 10 --remove-bpe \
	--batch-size 512

# echo "========experiment $N========" >> $OUTPUTS
tail -n 1 $GENERATE >> $OUTPUTS


# ==============================================
# train
# ==============================================
#####japaraで事前学習
# CUDA_VISIBLE_DEVICES=3 fairseq-train > $TRAIN 2> $TRAIN.error\
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
#     --no-epoch-checkpoints 

#####kyotoでファインチューニング
# CUDA_VISIBLE_DEVICES=3 fairseq-train > $TRAIN 2> $TRAIN.error\
#     $DATA\
#     --restore-file checkpoints/j-para/checkpoint_best.pt \
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
#     --max-epoch 10 \
#     --no-epoch-checkpoints 

