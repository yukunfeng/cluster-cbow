#!/usr/bin/env bash

set -x

log="log"

data_path="/home/lr/yukun/common_corpus/data/50lm/en"

# emb_path=/home/lr/yukun/large_scale_corpus/sample.middle.fasttext.emb.bin

# Google News
# emb_path=/home/lr/yukun/common_corpus/data/GoogleNews-vectors-negative300.bin
# python -u main.py --input_emb_path $emb_path --emb_format word2vec --tied --data $data_path --epoch 40 --emsize 300 >> $log

# Previous fasttext emb on wik only
# emb_path=/raid_elmo/home/lr/yukun/fasttext_embs/wiki.en.bin
# python -u main.py --input_emb_path $emb_path --emb_format fasttext --tied --data $data_path --epoch 40 --emsize 300 >> $log

# Previous fasttext emb on common crawl
# emb_path=/raid_elmo/home/lr/yukun/fasttext_embs/cc.en.300.bin
# python -u main.py --input_emb_path $emb_path --emb_format fasttext --tied --data $data_path --epoch 40 --emsize 300 >> $log

# cluster_nums=(3425 10286 20555)
# cluster_nums=(20555)
# for cluster_num in "${cluster_nums[@]}"
# do
    # emb_path=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf.fasttext.outfreq18.cluster${cluster_num}.emb.bin
    # python -u main.py --input_emb_path $emb_path --emb_format fasttext --tied --data $data_path --epoch 40 --emsize 300 >> $log
    # emb_path=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf.fasttext.outfreq50.cluster${cluster_num}.emb.bin
    # python -u main.py --input_emb_path $emb_path --emb_format fasttext --tied --data $data_path --epoch 40 --emsize 300 >> $log
# done

# emb_path=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf.fasttext.emb
# python -u main.py --input_emb_path $emb_path --emb_format fasttext --tied --data $data_path --epoch 40 --emsize 300 >> $log

emb_path=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf.fasttext.minc20.emb.bin
python -u main.py --input_emb_path $emb_path --emb_format fasttext --tied --data $data_path --epoch 40 --emsize 300 >> $log

# word-lm with random initialized word embeddings.
# python -u main.py --tied  --data $data_path --epoch 40 --emsize 300  >> $log
