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

# subword-cbow with word clusters
# cluster_nums=(3425 10286 20555)
# cluster_nums=(20555)
# for cluster_num in "${cluster_nums[@]}"
# do
    # input=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf
    # cluster="${input}.cluster.$cluster_num"
    # output="${input}.fasttext.outfreq18.cluster${cluster_num}.emb"
    # ./fasttext cbow -input $input -minCount 5 -dim 300 -thread 40 \
        # -output $output -epoch 10 -minn 3 -maxn 6 -neg 10 -loss ns \
        # -cluster $cluster -freq_thre_out 18 -lowersearch_cluster 
# done

input=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf
output="${input}.fasttext.minc18.emb"
./fasttext cbow -input $input -minCount 20 -dim 300 -thread 40 \
    -output $output -epoch 10 -minn 3 -maxn 6 -neg 10 -loss ns

