#!/usr/bin/env bash

set -x

# emb_path=/home/lr/yukun/large_scale_corpus/sample.middle.fasttext.emb.bin
# python ./emb_eval.py --input_emb_path $emb_path --emb_format 'fasttext'
emb_path=/home/lr/yukun/common_corpus/data/GoogleNews-vectors-negative300.bin
python ./emb_eval.py --input_emb_path $emb_path --emb_format 'word2vec'
