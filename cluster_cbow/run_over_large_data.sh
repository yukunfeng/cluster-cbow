#!/usr/bin/env bash

set -x

# input=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf

# Debug
input=/home/lr/yukun/large_scale_corpus/sample.middle
cluster="${input}.cluster.1600"
output="${input}.fasttext.emb"
gdb --args ./fasttext cbow -input $input -minCount 5 -dim 300 -thread 40 \
    -output $output -epoch 10 -minn 3 -maxn 6 -neg 10 -loss ns \
    -cluster $cluster -freq_thre_out 100 -lowersearch_cluster 

# subword-cbow without using word clusters.

# output="${input}.fasttext.emb"
# ./fasttext cbow -input $input -minCount 5 -dim 300 -thread 40 \
    # -output $output -epoch 10 -minn 3 -maxn 6 -neg 10 -loss ns