#!/usr/bin/env bash

set -x


# Debug
# input=/home/lr/yukun/large_scale_corpus/sample.middle
# cluster="${input}.cluster.1600"
# output="${input}.fasttext.emb"
# gdb --args ./fasttext cbow -input $input -minCount 5 -dim 300 -thread 40 \
    # -output $output -epoch 10 -minn 3 -maxn 6 -neg 10 -loss ns \
    # -cluster $cluster -freq_thre_out 100 -lowersearch_cluster 


# subword-cbow without using word clusters.

# input=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf
input=/home/lr/yukun/large_scale_corpus/sample.middle
output="${input}.fasttext.emb"
./fasttext cbow -input $input -minCount 5 -dim 300 -thread 40 \
    -output $output -epoch 10 -minn 3 -maxn 6 -neg 10 -loss ns

# subword-cbow with word clusters
# input=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf
# output="${input}.fasttext.outfreq50.emb"
# cluster="${input}.cluster.1600"
# ./fasttext cbow -input $input -minCount 5 -dim 300 -thread 40 \
    # -output $output -epoch 10 -minn 3 -maxn 6 -neg 10 -loss ns \
    # -cluster $cluster -freq_thre_out 50 -lowersearch_cluster 

# input=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf
# output="${input}.fasttext.outfreq30.emb"
# cluster="${input}.cluster.1600"
# ./fasttext cbow -input $input -minCount 5 -dim 300 -thread 40 \
    # -output $output -epoch 10 -minn 3 -maxn 6 -neg 10 -loss ns \
    # -cluster $cluster -freq_thre_out 30 -lowersearch_cluster 
