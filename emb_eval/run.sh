#!/usr/bin/env bash

set -x

# emb_path=/home/lr/yukun/common_corpus/data/GoogleNews-vectors-negative300.bin
# python ./emb_eval.py --input_emb_path $emb_path --emb_format 'word2vec'

# Previous fasttext emb on wik only
# emb_path=/raid_elmo/home/lr/yukun/fasttext_embs/wiki.en.bin
# python ./emb_eval.py --input_emb_path $emb_path --emb_format 'fasttext'

# Previous fasttext emb on common crawl
# emb_path=/raid_elmo/home/lr/yukun/fasttext_embs/cc.en.300.bin
# python ./emb_eval.py --input_emb_path $emb_path --emb_format 'fasttext'

# cluster_nums=(3425 10286 20555)
# for cluster_num in "${cluster_nums[@]}"
# do
    # emb_path=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf.fasttext.outfreq50.cluster${cluster_num}.emb.bin
    # python ./emb_eval.py --input_emb_path $emb_path --emb_format 'fasttext'
# done

# emb_path=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf.fasttext.outfreq18.cluster20555.emb.bin
# python ./emb_eval.py --input_emb_path $emb_path --emb_format 'fasttext'

emb_path=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf.fasttext.emb
python ./emb_eval.py --input_emb_path $emb_path --emb_format 'fasttext'
